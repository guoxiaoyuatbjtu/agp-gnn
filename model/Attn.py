import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils import weight_norm
from model.RBN import RepresentativeBatchNorm2d

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table, gain=np.sqrt(2.0))

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, bias=False, norm=True):
        super(FC, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                weight_norm(nn.Conv2d(input_dim, output_dim, 1, stride=stride, bias=bias)),
                RepresentativeBatchNorm2d(output_dim),
            )
        else:
            self.conv = nn.Sequential(
                weight_norm(nn.Conv2d(input_dim, output_dim, 1, stride=stride, bias=bias)),
            )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = int(d_model / n_heads)
        self.n_heads = n_heads
        self.W_Q = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
        self.W_K = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
        self.W_V = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
        self.fc = FC(d_model, d_model, norm=False)

        # self.norm = nn.BatchNorm2d(d_model)
        self.norm = RepresentativeBatchNorm2d(d_model)

        self.relative_position_k = RelativePosition(self.d_k, 11)
        self.relative_position_v = RelativePosition(self.d_k, 11)

        self.w1 = nn.Parameter(torch.zeros(n_heads, n_heads))
        self.w2 = nn.Parameter(torch.zeros(n_heads, n_heads))
        nn.init.xavier_uniform_(self.w1.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.w2.data, gain=np.sqrt(2.0))

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.ReLU()
    
    
    def forward(self, input_Q, input_K, input_V, attn_mask=None, attn_pre=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v, d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        len_q, len_k, len_v = input_Q.shape[2], input_K.shape[2], input_V.shape[2]
        residual, batch_size, num_nodes = input_V, input_V.size(0), input_V.shape[1]
        # print('Attn [0]', input_Q.shape, input_V.shape, input_K.shape)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q)
        # print('Attn [1]', Q.shape)
        Q_1 = Q.view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V)
        V_1 = V.view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # V: [batch_size, n_heads, len_v(=len_k), d_k]
        # print('Attn [2]', Q_1.shape, K.shape, V_1.shape)
        scores1 = torch.matmul(Q_1, K.transpose(-1, -2))
        # print('Attn [3]', scores1.shape)
        Q_2 = Q.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*num_nodes*self.n_heads, self.d_k)
        # print('Attn [4]', Q_2.shape)
        K_2 = self.relative_position_k(len_q, len_k)   #len_q, len_k, d_k
        # print('Attn [5]', K_2.shape)
        scores2 = torch.matmul(Q_2, K_2.transpose(1, 2)).transpose(0, 1)
        # print('Attn [6]', scores2.shape)
        scores2 = scores2.contiguous().view(batch_size, num_nodes, self.n_heads, len_q, len_k)
        scores = (scores1 + scores2) / np.sqrt(self.d_k)
        del scores1, scores2
        # print('Attn [7]', scores.shape, attn_mask.shape)
        if attn_mask != None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(-1, num_nodes, self.n_heads, -1, -1)
            scores.masked_fill_(attn_mask, -1e9)

        scores = scores.permute(0, 1, 3, 4, 2) @ self.w1
        scores = scores.permute(0, 1, 4, 2, 3)
        attn = nn.Softmax(dim=-1)(self.relu(scores))   # barch, nhead, seq_len, seq_len
        # print('Attn [8]', scores.shape)
        # attn = self.dropout(attn)
        attn = attn.permute(0, 1, 3, 4, 2) @ self.w2
        attn = attn.permute(0, 1, 4, 2, 3).contiguous()

        if attn_pre != None:
            attn = attn_pre + attn
        # print('Attn [9]', attn.shape, V_1.shape)
        context_1 = torch.matmul(attn, V_1) # [batch_size, n_heads, len_q, d_v]
        # print('Attn [10]', context_1.shape)
        r_v2 = self.relative_position_v(len_q, len_v)
        # print('Attn [11]', r_v2.shape)
        attn = attn.permute(3, 0, 1, 2, 4).contiguous().view(len_q, batch_size*num_nodes*self.n_heads, len_k)
        # print('Attn [12]', attn.shape)
        context_2 = torch.matmul(attn, r_v2)
        # print('Attn [13]', context_2.shape)
        context_2 = context_2.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.n_heads, len_q, self.d_k)
        # print('Attn [14]', context_2.shape, context_1.shape)
        context = context_1 + context_2
        context = context.transpose(2, 3).reshape(batch_size, num_nodes, -1, self.n_heads * self.d_k) # context_1: [batch_size, len_q, n_heads * d_v]
        # print('Attn [15]', context.shape)
        output = self.fc(self.norm(context.permute(0,3,1,2)).permute(0,2,3,1).contiguous())
        output = torch.relu(output) + residual
        # print('Attn [16]', output.shape, attn.shape)
        return output, attn

# class NormMultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads):
#         super(NormMultiHeadAttention, self).__init__()
#         self.d_k = int(d_model / n_heads)
#         self.n_heads = n_heads
#         self.W_Q = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
#         self.W_K = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
#         self.W_V = FC(d_model, self.d_k * n_heads, bias=False, norm=False)
#         self.fc = FC(d_model, d_model, norm=False)

#         # self.norm = nn.BatchNorm2d(d_model)
#         self.norm = RepresentativeBatchNorm2d(d_model)

#         self.relative_position_k = RelativePosition(self.d_k, 11)
#         self.relative_position_v = RelativePosition(self.d_k, 11)

#         self.w1 = nn.Parameter(torch.zeros(n_heads, n_heads))
#         self.w2 = nn.Parameter(torch.zeros(n_heads, n_heads))
#         nn.init.xavier_uniform_(self.w1.data, gain=np.sqrt(2.0))
#         nn.init.xavier_uniform_(self.w2.data, gain=np.sqrt(2.0))

#         self.dropout = nn.Dropout(0.1)
#         self.relu = nn.LeakyReLU(0.2)
    
    
#     def forward(self, input_Q, input_K, input_V, attn_mask=None, attn_pre=None):
#         '''
#         input_Q: [batch_size, len_q, d_model]
#         input_K: [batch_size, len_k, d_model]
#         input_V: [batch_size, len_v, d_model]
#         attn_mask: [batch_size, seq_len, seq_len]
#         '''
#         len_q, len_k, len_v = input_Q.shape[2], input_K.shape[2], input_V.shape[2]
#         residual, batch_size, num_nodes = input_V, input_V.size(0), input_V.shape[1]
#         # print('Attn [0]', input_Q.shape, input_V.shape, input_K.shape)
#         # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         Q = self.W_Q(input_Q)
#         # print('Attn [1]', Q.shape)
#         Q_1 = Q.view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # Q: [batch_size, n_heads, len_q, d_k]
#         K = self.W_K(input_K).view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # K: [batch_size, n_heads, len_k, d_k]
#         V = self.W_V(input_V)
#         V_1 = V.view(batch_size, num_nodes, -1, self.n_heads, self.d_k).transpose(2,3)  # V: [batch_size, n_heads, len_v(=len_k), d_k]
#         # print('Attn [2]', Q_1.shape, K.shape, V_1.shape)
#         scores1 = torch.matmul(Q_1, K.transpose(-1, -2))
#         scores = (scores1) / np.sqrt(self.d_k)

#         if attn_mask != None:
#             attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(-1, num_nodes, self.n_heads, -1, -1)
#             scores.masked_fill_(attn_mask, -1e9)

#         scores = scores.permute(0, 1, 3, 4, 2) @ self.w1
#         scores = scores.permute(0, 1, 4, 2, 3)
#         attn = nn.Softmax(dim=-1)(self.relu(scores))   # barch, nhead, seq_len, seq_len
#         attn = attn.permute(0, 1, 3, 4, 2) @ self.w2
#         attn = attn.permute(0, 1, 4, 2, 3).contiguous()

#         if attn_pre != None:
#             attn = attn_pre + attn
#         context_1 = torch.matmul(attn, V_1) # [batch_size, n_heads, len_q, d_v]
#         context = context_1
#         context = context.transpose(2, 3).reshape(batch_size, num_nodes, -1, self.n_heads * self.d_k) # context_1: [batch_size, len_q, n_heads * 
#         output = self.fc(self.norm(context.permute(0,3,1,2)).permute(0,2,3,1).contiguous())
#         output = torch.relu(output) + residual
#         return output, attn

