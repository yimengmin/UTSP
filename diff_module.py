import scipy.sparse as sp
import torch
import sys
import pickle
import networkx as nx
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import numpy as np
import json
from scipy.sparse import coo_matrix
from utils import sparse_mx_to_torch_sparse_tensor
#ref: https://discuss.pytorch.org/t/creating-a-sparse-tensor-from-csr-matrix/13658/4

def GCN_diffusion(sptensor,order,feature,device='cuda'):
    """
    Creating a normalized adjacency matrix with self loops.
    sptensor = W
    https://arxiv.org/pdf/1609.02907.pdf
    """
    I_n = torch.eye(sptensor.size(0))
    I_n = I_n.to(device)
    A_gcn = sptensor +  I_n
    degrees = torch.sum(A_gcn,0)
    D = degrees
    D = torch.pow(D, -0.5)
    D = D.unsqueeze(dim=1)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    for i in range(order):
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        A_gcn_feature = torch.matmul(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list



