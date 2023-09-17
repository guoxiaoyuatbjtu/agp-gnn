import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from model.Attn import MultiHeadAttention, FC
from model.GPR import GPRGNN

class Fusion(nn.Module):
    def __init__(self, d_model):
        super(Fusion, self).__init__()
        self.temporal = nn.Sequential(
            FC(d_model, d_model, bias=True, norm=True),
        )
        self.spatial = nn.Sequential(
            FC(d_model, d_model, bias=True, norm=True),
        )
        self.fusion = nn.Sequential(
            FC(d_model, d_model, bias=True, norm=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x_t, x_s):
        x_t_new = self.temporal(x_t)
        x_s_new = self.spatial(x_s)
        gate = torch.sigmoid(self.fusion(x_t + x_s))
        out = gate * x_t_new + (1-gate) * x_s_new
        return self.relu(out)
        

class STBlock(nn.Module):
    def __init__(self, args, d_model, n_heads, num_timesteps, idx):
        super(STBlock, self).__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.n_heads = n_heads
        self.idx = idx

        self.multiHeadAttention = MultiHeadAttention(d_model, n_heads)

        self.prop = GPRGNN(args, 2, d_model)
        self.embed = nn.Sequential(
            FC(d_model*2, d_model, bias=True),
            nn.ReLU(),
        )
        self.fusion = Fusion(d_model)
        

    def forward(self, x, DTE, adj, attn_shape=None, temporal_attn=None):

        res = x
        x = torch.cat([x, DTE], dim=3)
        x = self.embed(x)
        temporal, temporal_attn = self.multiHeadAttention(x, x, x, attn_shape, temporal_attn)
        spatial = self.prop(x, DTE, adj)
        x = self.fusion(temporal, spatial) + res
        return x, temporal_attn

    def annealing(self):
        self.prop.annealing()


class EndFC(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(EndFC, self).__init__()
        self.fc1 = nn.Sequential(
            FC(in_feature, 4*in_feature, bias=True),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            FC(4*in_feature, 4*in_feature, bias=True),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            FC(4*in_feature, out_feature, bias=True, norm=False),
        )
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x) + x
        x = self.fc3(x)

        return x


class AGPGNN(nn.Module):
    def __init__(self, args, cuda, num_nodes, num_features, DE_features, TE_features, num_timesteps_input,
                 num_timesteps_output, L=3, nheads=8, nhid=64, layers=4, dropout=0.6, alpha=0.2):
        super(AGPGNN, self).__init__()
        self.cuda_device = cuda
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.num_timesteps_output = num_timesteps_output
        self.num_timesteps_input = num_timesteps_input
        self.nhid = nhid
        self.num_features = num_features
        self.DE_features = DE_features
        self.TE_features = TE_features
        self.nheads = nheads
        self.L = L

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, 32))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_in_day_emb = nn.Parameter(torch.empty(288, 32))
        nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = nn.Parameter(torch.empty(7, 32))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.DTE_features = DE_features+TE_features
        self.start = nn.Sequential(
             FC(num_features + 32 * 3, 2 * self.nhid, bias=True),
             nn.ReLU(),
             FC(2 * self.nhid, self.nhid, bias=True),
             nn.ReLU(),
        )
        self.DTE_embed = nn.Sequential(
            FC(self.DTE_features, 2 * self.nhid, bias=True),
            nn.ReLU(),
            FC(2 * self.nhid, self.nhid, bias=True),
            nn.ReLU(),
        )

        self.STBlocks1 = nn.ModuleList([STBlock(args, self.nhid, nheads, num_timesteps_input, i) for i in range(L)])

        self.transformsAttn = MultiHeadAttention(self.nhid, nheads)

        self.STBlocks2 = nn.ModuleList([STBlock(args, self.nhid, nheads, num_timesteps_input, i+L) for i in range(L)])

        self.fc = nn.Sequential(
            FC(self.nhid, 4 * self.nhid, bias=True),
            nn.ReLU(),
            FC(4 * self.nhid, 1, bias=True, norm=False)
        )


    def forward(self, x, t_, d_, n_, DE, TE):

        t_ = self.time_in_day_emb[t_[:, -1, :]]
        t_ = t_.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        
        d_ = self.day_in_week_emb[d_[:, -1, :]]
        d_ = d_.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        
        n_ = self.node_emb[n_[:, -1, :]]
        n_ = n_.unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        TE_pred = TE[:,:,-self.num_timesteps_output:,:]
        TE = TE[:,:,:self.num_timesteps_input,:]

        DE = DE.unsqueeze(2).expand(-1, -1, self.num_timesteps_input, -1)
        DTE_his = torch.cat([DE, TE], dim=3)
        DTE_pred = torch.cat([DE, TE_pred], dim=3)

        attn_mask = get_attn_mask([x.shape[0], self.num_timesteps_input, self.num_timesteps_output]).to(x.device)
        
        x = torch.cat([x, t_, d_, n_], dim=3)
        x = self.start(x)

        DTE_his = self.DTE_embed(DTE_his)
        DTE_pred = self.DTE_embed(DTE_pred)
        temporal_attn = None
        res_list = []
        for unit in range(self.L):
            temporal_attn = None
            x, temporal_attn = self.STBlocks1[unit](x, DTE_his, attn_mask, temporal_attn)
            res_list.append(x)
        
        x, transAttn = self.transformsAttn(DTE_pred, DTE_his, x)
        
        temporal_attn = None
        for unit in range(self.L):
            temporal_attn = None
            x, temporal_attn = self.STBlocks2[unit](x, DTE_pred, attn_mask, temporal_attn)
            x = x + res_list[self.L-unit-1]
        
        out = self.fc(x)
        out = out.view(-1, self.num_nodes, self.num_timesteps_output, 1).permute(0, 3, 1, 2)

        return out

    def annealing(self):
        for unit in range(self.L):
            self.STBlocks1[unit].annealing()
            self.STBlocks2[unit].annealing()


def get_attn_mask(attn_shape):
    attn_mask = torch.triu(torch.ones(attn_shape), 1) # Upper triangular matrix
    attn_mask = attn_mask.bool()
    return attn_mask
