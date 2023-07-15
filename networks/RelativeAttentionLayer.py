import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePosition(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, pos_mat):
        """pos_mat: relative position matrix, n * n * pos_dim"""
        assert pos_mat.shape[0] == pos_mat.shape[1]
        # all seq share one relative positional matrix
        n_element = pos_mat.shape[0]
        pos_dim = pos_mat.shape[-1]
        positions = pos_mat.view(-1, pos_dim)
        pos_embeddings = self.linear_2(self.linear_1(positions))

        # [sz_b x len_q x len_q x d_v/d_k]
        return pos_embeddings.view(n_element, n_element, -1)  # added: batch_size dim


# ----------------------- New versions of relative position -----------------------
class NewRelativeMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, pos_dim, temperature, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.relative_position = RelativePosition(pos_dim, d_k, d_k)
        self.attention = RelativeScaledDotProductAttention(temperature=temperature)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, pos_mat, mask=None):
        d_k, d_v, d_model, n_head = self.d_k, self.d_v, self.d_model, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # generate the spatial relative position embeddings (SRPEs)
        a_k = self.relative_position(pos_mat)

        if mask is not None:  # used to achieve Shielded Attention
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, a_k, d_k, d_v, n_head, mask=mask)

        # Transpose to move the head dimension back: sz_b x len_q x n_head x dv
        # Combine the last two dimensions to concatenate all the heads together: sz_b x len_q x (n_head*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class RelativeScaledDotProductAttention(nn.Module):
    ''' attn: sum over element-wise product of three vectors'''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, a_k, d_k, d_v, n_head, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Transpose for attention dot product: sz_b x n_head x len_q x dv
        # Separate different heads: sz_b x len_q x n_head x dv
        r_q1, r_k1, r_v1 = q.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           k.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           v.view(sz_b, len_v, n_head, d_v).permute(0, 2, 1, 3)

        # r_q1: [sz_b, n_head, len_q, 1, d_k], r_k1: [sz_b, n_head, 1, len_q, d_k]
        attn1 = torch.mul(r_q1.unsqueeze(2), r_k1.unsqueeze(3))
        # attn1: [sz_b, n_head, len_q, len_q, d_k], a: [len_q, len_q, d_k]
        attn = torch.sum(torch.mul(attn1, a_k), -1)
        attn = attn / self.temperature  # [sz_b x n_head x len_q x len_k]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, r_v1)

        return output, attn


if __name__ == "__main__":
    pass

