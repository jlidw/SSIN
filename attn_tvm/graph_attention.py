import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import numpy as np

import sys
sys.path.append('..')  # import the upper directory of the current file into the search path
from SSIN.attn_tvm.three_tensor_mm_tvm import graph_three_vec_mm as QKP_mm_tvm
from SSIN.attn_tvm.two_tensor_mm_tvm import graph_two_vec_mm as AttnV_mm_tvm
import pynvml


def generate_attn_mask(seq_len, station_num):
    """ attn_mask is also the q_k_mask, it is a n*n matrix.
        attn_mask[i,j] = 0 means NO edge between q_i and k_j;
        attn_mask[i,j] = 1 means having edge between q_i and k_j. """
    attn_mask = np.zeros((seq_len, seq_len))  # n*n matrix
    unmasked_indexes = range(0, station_num)

    # for each node, cut off the edge between it and invalid (masked or padded) nodes
    attn_mask[:, unmasked_indexes] = 1
    # for each node (mainly for masked nodes), add itself edge for attention calculation
    eye_mask = np.eye(seq_len)

    attn_mask = np.logical_or(attn_mask, eye_mask).astype(int)
    attn_mask = np.expand_dims(attn_mask, 0)  # add a batch dim

    attn_mask = torch.IntTensor(attn_mask)
    return attn_mask


def get_k_q_mask(seq_len, station_num):
    """Get the key-query index from query-key index for any directed or undirected graphs
    k_q_mask: n*n matrix, in case that one node is the neighbor for all other nodes.
              k_q_mask[b,i,j]= 0 means k_i is NOT neighbor of q_j;
              k_q_mask[b,i,j]= 1 means k_i is a neighbor of q_j;"""
    k_q_mask = np.zeros((seq_len, seq_len))  # n*n matrix
    unmasked_indexes = range(0, station_num)

    # for each node, cut off the edge between it and invalid (masked or padded) nodes
    k_q_mask[unmasked_indexes, :] = 1
    # for each node (mainly for masked nodes), add itself edge for attention calculation
    eye_mask = np.eye(seq_len)

    k_q_mask = np.logical_or(k_q_mask, eye_mask).astype(int)
    k_q_mask = np.expand_dims(k_q_mask, 0)  # add a batch dim

    k_q_mask = torch.IntTensor(k_q_mask)
    return k_q_mask


"""Naive implementation: Multi-head shielded self-attention with RPE"""
class NormalShieldedAttentionWithRPE(nn.Module):
    def __init__(self, opt):
        super(NormalShieldedAttentionWithRPE, self).__init__()
        self.n_head = opt.n_head
        self.d_k = opt.d_k

        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)

        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)

        self.seq_len = opt.seq_len
        self.station_num = opt.station_num
        self.mask = generate_attn_mask(self.seq_len, self.station_num).cuda()

    def forward(self, hidden_states, rpe):
        """rpe: relative position embedding"""
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        q = self.w_qs(hidden_states)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)

        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        attn_weights_1 = torch.mul(q.unsqueeze(2), k.unsqueeze(3))  # attn1: [sz_b, n_head, len_q, len_q, d_k]
        attn_weights = torch.sum(torch.mul(attn_weights_1, rpe.unsqueeze(1)), -1)  # [sz_b x n_head x len_q x len_k]
        attn_weights = attn_weights / (self.d_k ** 0.5)

        if self.mask is not None:
            attn_weights = attn_weights.masked_fill(self.mask.unsqueeze(1) == 0, -1e10)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        context = self.dropout_fc(self.fc(attn))
        context += residual

        context = self.layer_norm(context)

        return context


"""TVM implementation: Multi-head shielded self-attention with RPE"""
class TVMShieldedAttentionWithRPE(nn.Module):
    def __init__(self, opt):
        super(TVMShieldedAttentionWithRPE, self).__init__()
        self.n_head = opt.n_head
        self.d_k = opt.d_k

        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)

        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)

        self.seq_len = opt.seq_len
        self.station_num = opt.station_num
        self.q_k_mask = generate_attn_mask(self.seq_len, self.station_num).cuda()
        self.k_q_mask = get_k_q_mask(self.seq_len, self.station_num).cuda()

    def forward(self, hidden_states, rpe):
        """rpe: relative position embedding"""
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        q = self.w_qs(hidden_states)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)

        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        attn_weights = QKP_mm_tvm(q, k, rpe, self.q_k_mask, self.k_q_mask, False, -1000000000)
        attn_weights = attn_weights / (self.d_k ** 0.5)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        attn = AttnV_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.contiguous().view(bsz, seq_len, -1)

        context = self.dropout_fc(self.fc(attn))
        context += residual

        context = self.layer_norm(context)

        return context


"""Testing functions"""
def test_NSA(args, out_path):
    """Test the time and CUDA memory consumption of normal shielded self-attention with RPE."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu_id))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # bit
    init_mem = meminfo.used / 1024 ** 3

    NSA_Layer = NormalShieldedAttentionWithRPE(args).cuda()

    hidden_state = torch.rand(1, args.seq_len, args.d_model, dtype=torch.float32).cuda()
    rpe = torch.rand(1, args.seq_len, args.seq_len, args.d_k, dtype=torch.float32).cuda()

    used_memory = 0
    start_time = time.time()
    for i in range(args.repeat_times):
        NSA_Layer.eval()
        with torch.no_grad():
            result = NSA_Layer(hidden_state, rpe)
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu_id))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_memory += meminfo.used / 1024 ** 3

    used_avg_time = round((time.time() - start_time) / args.repeat_times * 1000, 4)
    used_avg_memory = round(used_memory / args.repeat_times - init_mem, 4)

    with open(out_path, 'a') as f:
        f.writelines(f"Seq_length: {args.seq_len}, "
                     f"Used average time: {used_avg_time} ms, "
                     f"Used average memory: {used_avg_memory} GB" + '\n')


def test_TSA(args, out_path):
    """Test the time and CUDA memory consumption of TVM-version shielded self-attention with RPE;
       the testing is built on the inference stage, so no backward is needed."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu_id))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024 ** 3

    TSA_Layer = TVMShieldedAttentionWithRPE(args).cuda()

    hidden_state = torch.rand(1, args.seq_len, args.d_model, dtype=torch.float32).cuda()
    rpe = torch.rand(1, args.seq_len, args.seq_len, args.d_k, dtype=torch.float32).cuda()

    used_memory = 0
    start_time = time.time()
    for i in range(args.repeat_times):
        TSA_Layer.eval()
        with torch.no_grad():
            result = TSA_Layer(hidden_state, rpe)
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu_id))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_memory += meminfo.used / 1024 ** 3

    used_avg_time = round((time.time() - start_time) / args.repeat_times * 1000, 4)
    used_avg_memory = round(used_memory / args.repeat_times - init_mem, 4)

    with open(out_path, 'a') as f:
        f.writelines(f"Seq_length: {args.seq_len}, "
                     f"Used average time: {used_avg_time} ms, "
                     f"Used average memory: {used_avg_memory} GB" + '\n')


def parsing():
    parser = argparse.ArgumentParser(description='Needed for graph self attention.')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--seq_len', type=int, default=5000)
    parser.add_argument('--station_num', type=int, default=123)
    parser.add_argument('--pos_dim', type=int, default=2)  # relative position: [distance, azimuth]

    parser.add_argument('--repeat_times', type=int, default=1000)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--test_model", type=str, default="NSA")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--out_name", type=str, default="efficiency.txt")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parsing()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if torch.cuda.is_available():
        print('Using device: {}'.format(torch.cuda.get_device_name()))

    out_dir = args.out_dir
    out_name = args.out_name
    test_model = args.test_model
    gpu_id = args.gpu_id

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{test_model}_{out_name}"

    if test_model == "NSA":
        test_NSA(args, out_path)
    elif test_model == "TSA":
        test_TSA(args, out_path)


