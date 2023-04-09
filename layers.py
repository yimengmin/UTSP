import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import sparse_mx_to_torch_sparse_tensor
class GC(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """
    def __init__(self, in_features, out_features):
        super(GC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Linear(in_features, out_features)
    def forward(self, input, adj,device='cuda'):
        # adj is extracted from the graph structure
        support = self.mlp(input)

#        adj = adj.to_torch_sparse_coo_tensor()
#        support = torch.mm(input, self.weight)
        I_n = sp.eye(adj.size(0))
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        A_gcn = adj +  I_n
        degrees = torch.sparse.sum(A_gcn,0)
        D = degrees
        D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        output = torch.mul(A_gcn_feature,D)
        return output

class GC_DenseA(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """
    def __init__(self, in_features, out_features):
        super(GC_DenseA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Linear(in_features, out_features)
    def forward(self, input, adj,device='cuda'):
        # adj is extracted from the graph structure
        support = self.mlp(input)

#        adj = adj.to_torch_sparse_coo_tensor()
#        support = torch.mm(input, self.weight)
        I_n = torch.eye(adj.size(0)).to(device)
        A_gcn = adj +  I_n
        degrees = torch.sum(A_gcn,0)
        D = degrees
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        A_gcn_feature = torch.matmul(A_gcn,A_gcn_feature)
        output = torch.mul(A_gcn_feature,D)
        return output

class GC_withres(Module):
    """
    res conv
    """
    def __init__(self, in_features, out_features,smooth):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
#        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#        if bias:
#            self.bias = Parameter(torch.FloatTensor(out_features))
#        else:
#            self.register_parameter('bias', None)
#        self.reset_parameters()
        self.mlp = nn.Linear(in_features, out_features)
#    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
#        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
#        if self.bias is not None:
#            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj,device='cuda'):
        # adj is extracted from the graph structure
#        adj = adj.to_torch_sparse_coo_tensor()
#        support = torch.mm(input, self.weight)
        support = self.mlp(input)
        I_n = sp.eye(adj.size(0))
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
        A_gcn = adj +  I_n
        degrees = torch.sparse.sum(A_gcn,0)
        D = degrees
        D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)
        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        output = A_gcn_feature * self.smooth + support
        output = output/(1+self.smooth)
        return output

