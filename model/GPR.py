import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attn import FC


class EmbeddingHead(nn.Module):
    def __init__(self, args, input_dim, d_model, K, embedding_dim):
        super(EmbeddingHead, self).__init__()
        self.E = nn.Parameter(torch.zeros(K, embedding_dim))
        nn.init.xavier_uniform_(self.E.data, gain=np.sqrt(2.0))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, K),
            nn.Linear(K, K),
            nn.Softplus(),
        )
        self.nodes_num = args.num_of_vertices

    def forward(self, x):
        x = x.view(x.shape[0], self.nodes_num, -1)
        C = self.fc(x)
        D = F.gumbel_softmax(C, tau=1, hard=False, dim=-1)
        Embedd = D @ self.E
        return Embedd


class CompressingEmbedding(nn.Module):
    def __init__(self, args, input_dim, d_model, M=8, K=64, embedding_dim=64):
        super(CompressingEmbedding, self).__init__()
        self.modules_list = nn.ModuleList()
        self.A = nn.Parameter(torch.zeros(M*K, embedding_dim))
        nn.init.xavier_uniform_(self.A.data, gain=np.sqrt(2.0))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, M*K//2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(M*K//2, M*K),
            nn.ReLU(),
        )
        self.nodes_num = args.num_of_vertices
        self.M = M
        self.K = K
        self.tau = 10

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, self.nodes_num, -1)
        x = self.fc(x).view(batch, self.nodes_num, self.M, self.K)
        x = F.gumbel_softmax(x, tau=self.tau, dim=-1).view(batch, self.nodes_num, self.M * self.K)
        E = torch.matmul(x, self.A)
        return E
            
    def annealing(self):
        self.tau = self.tau * 0.99999

class GPR_prop(torch.nn.Module):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha=0.1):
        super(GPR_prop, self).__init__()
        self.K = K
        self.alpha = alpha
        bound = np.sqrt(3/(K+1))
        TEMP = np.random.uniform(-bound, bound, K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.temp = nn.Parameter(torch.FloatTensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
    
    def forward(self, x, A_hat):
        batch, num_nodes, num_timesteps, d_model = x.size()
        x = x.view(x.shape[0], x.shape[1], -1)
        output = self.temp[0] * x
        for i in range(self.K):
            x = A_hat @ x
            output = output + self.temp[i+1] * x
        output = output.view(batch, num_nodes, num_timesteps, d_model)
        return output


class GPRGNN(torch.nn.Module):
    def __init__(self, args, K, d_model):
        super(GPRGNN, self).__init__()
        self.linear1 = nn.Sequential(
            FC(d_model, d_model, bias=True, norm=True),
        )
        self.prop = GPR_prop(K)
        self.d_model = d_model
        self.nodes_num = args.num_of_vertices
        

        self.E = nn.Parameter(torch.zeros(args.num_of_vertices, d_model))
        nn.init.xavier_uniform_(self.E.data, gain=np.sqrt(2.0))

        self.prop.reset_parameters()
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.h = nn.Parameter(torch.randn((1)))
        self.bn = nn.BatchNorm1d(args.num_of_vertices)
        
    
    def forward(self, x):
        A_hat = torch.softmax(self.relu(self.E @ self.E.T), dim=-1) + torch.eye(self.nodes_num, device=x.device)
        res = x
        x = self.linear1(x)
        x = self.prop(x, A_hat)
        x = torch.relu(x) + res
        return x

    def annealing(self):
        return
        self.CE.annealing()
