import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import numpy as np

import sys
sys.path.append('../..')  # import the upper directory of the current file into the search path
from SSIN.attn_tvm.three_tensor_mm_tvm import graph_three_vec_mm as QKP_mm_tvm
from SSIN.attn_tvm.two_tensor_mm_tvm import graph_two_vec_mm as AttnV_mm_tvm


"""TVM implementation: Multi-head shielded self-attention with RPE"""


class TVMShieldedAttentionWithRPE(nn.Module):
    def __init__(self, opt, q_k_mask, k_q_mask):
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
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

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

