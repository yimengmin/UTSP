import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
import torch.nn
import torch_geometric.utils as sp
from torch_geometric.data import Data
from torch.nn import Parameter
from diff_module import GCN_diffusion, scattering_diffusion
from diff_moduleS4 import scattering_diffusionS4
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from layers import GC_withres, GC,GC_DenseA
import torch.nn as nn

class SCTConv(torch.nn.Module):

    def __init__(self, hidden_dim, smooth, dropout, Withgres=False):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        self.smoothlayer = Withgres
        self.gres = GC_withres(hidden_dim, hidden_dim, smooth=smooth)
        self.dropout = dropout

    def forward(self, X, adj, moment=1, device='cuda'):
        """
        Params
        ------
        adj [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X
        N = support0.size()[0]
        h = support0
        gcn_diffusion_list = GCN_diffusion(adj, 3, support0, device=device)
        h_A = gcn_diffusion_list[0]
        h_A2 = gcn_diffusion_list[1]
        h_A3 = gcn_diffusion_list[2]
        h_A = nn.LeakyReLU()(h_A)
        h_A2 = nn.LeakyReLU()(h_A2)
        h_A3 = nn.LeakyReLU()(h_A3)
#        h_sct1, h_sct2, h_sct3 = scattering_diffusion(adj, support0)
# S4
        h_sct1, h_sct2, h_sct3, h_sct4 = scattering_diffusionS4(adj, support0)

        h_sct1 = torch.abs(h_sct1) ** moment
        h_sct2 = torch.abs(h_sct2) ** moment
        h_sct3 = torch.abs(h_sct3) ** moment
# S4
        h_sct4 = torch.abs(h_sct4) ** moment


        a_input_A = torch.hstack((h, h_A)).unsqueeze(1)
        a_input_A2 = torch.hstack((h, h_A2)).unsqueeze(1)
        a_input_A3 = torch.hstack((h, h_A3)).unsqueeze(1)
        a_input_sct1 = torch.hstack((h, h_sct1)).unsqueeze(1)
        a_input_sct2 = torch.hstack((h, h_sct2)).unsqueeze(1)
        a_input_sct3 = torch.hstack((h, h_sct3)).unsqueeze(1)
# S4
        a_input_sct4 = torch.hstack((h, h_sct4)).unsqueeze(1)


# S3
#        a_input = torch.cat((a_input_A, a_input_A2, a_input_A3, a_input_sct1, a_input_sct2, a_input_sct3), 1).view(N, 6, -1)
# S4
#        a_input = torch.cat((a_input_A, a_input_A2, a_input_A3, a_input_sct1, a_input_sct2, a_input_sct3,a_input_sct4), 1).view(N, 7, -1)
# 1 low pass + 3 high pass
        a_input = torch.cat((a_input_A, a_input_A2,a_input_sct1, a_input_sct2, a_input_sct3,a_input_sct4), 1).view(N, 6, -1)

        e = torch.matmul(torch.nn.functional.relu(a_input), self.a).squeeze(2)
# S3
#        attention = F.softmax(e, dim=1).view(N, 6, -1)
# S4
        attention = F.softmax(e, dim=1).view(N, 6, -1)
# S3
#        h_all = torch.cat((h_A.unsqueeze(dim=2), h_A2.unsqueeze(dim=2), h_A3.unsqueeze(dim=2),
#        h_sct1.unsqueeze(dim=2), h_sct2.unsqueeze(dim=2), h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
# S4
        h_all = torch.cat((h_A.unsqueeze(dim=2), h_A2.unsqueeze(dim=2),h_sct1.unsqueeze(dim=2), h_sct2.unsqueeze(dim=2), h_sct3.unsqueeze(dim=2), h_sct4.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        h_prime = torch.mul(attention, h_all)
        h_prime = torch.mean(h_prime, 1)
        if self.smoothlayer:
            h_prime = self.gres(h_prime, adj, device)
        else:
            X = self.linear1(h_prime)
            X = F.leaky_relu(X)
            X = self.linear2(X)
            X = F.leaky_relu(X)
            return X


class GNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, Withgres=False, smooth=0.0):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm1d(input_dim)
        self.dropout = dropout
        self.smooth = smooth
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim, self.smooth, self.dropout, Withgres))

        self.mlp1 = Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2 = Linear(hidden_dim, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.m = torch.nn.Softmax(dim=0)
#        self.mlp1 = Linear(hidden_dim * (1 + n_layers), output_dim)

    def forward(self, X, adj, moment=1, device='cuda'):
#        numnodes = X.size(0)
#        scale = np.sqrt(numnodes)
#        X = X / scale
        X = self.bn0(X)
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X, adj, moment=moment, device=device)
#            X = self.bn1(X)
#            X = X / scale
            hidden_states = torch.cat([hidden_states, X], dim=1)

        X = hidden_states
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X)
        X = self.m(X)
#        X = F.relu(X)
#        X = F.sigmoid(X)
#        X = X/torch.sum(X,1).unsqueeze(1)
        return X


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GC_DenseA(input_dim, hidden_dim)
        self.gc2 = GC_DenseA(hidden_dim, hidden_dim)
        self.mlp2 = Linear(hidden_dim, output_dim)
        self.m = torch.nn.Softmax(dim=0)

    def forward(self, X, A, device='cuda'):
        X = F.leaky_relu(self.gc1(X, A, device))
        X = F.dropout(X, (self.dropout), training=(self.training))
        X = self.gc2(X, A, device)
        X = F.leaky_relu(X)
        X = self.mlp2(X)
        X = self.m(X)
        return X


