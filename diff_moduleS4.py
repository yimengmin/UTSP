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
#    I_n = sparse_mx_to_torch_sparse_tensor(I_n).to(device)
    A_gcn = sptensor +  I_n
#    print('Type of A_gcn')
#    print(type(A_gcn))
#    degrees = torch.sparse.sum(A_gcn,0)
    degrees = torch.sum(A_gcn,0)
    D = degrees
#    D = D # transfer D from sparse tensor to normal torch tensor
    D = torch.pow(D, -0.5)
    D = D.unsqueeze(dim=1)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    for i in range(order):
#        print('GCN diffusion step: %d'%i)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
#        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.matmul(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list

def SCT1st(sptensor,order,feature):
    '''
    sptensor = W
    '''
    degrees = torch.sum(sptensor,0)
    D = degrees
#    D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=1)
    iteration = 2**(order-1)
    feature_p = feature
    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.matmul(sptensor,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
#        feature_p = torch.spmm(adj_sct,feature_p) #compute P^{2^(k-1)}
    featura_loc = feature_p
    for j in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.matmul(sptensor,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
#        feature_p = torch.spmm(adj_sct,feature_p) #compute P^{2^k}
    feature_p = featura_loc - feature_p
    return feature_p

def scattering_diffusionS4(sptensor,feature):
    '''
    A_tilte,adj_p,shape(N,N)
    feature:shape(N,3) :torch.FloatTensor
    all on cuda
    '''
    #generate 1st scattering feature
#    h_sct1 = SCT1st(sptensor,1,feature)
#    h_sct2 = SCT1st(sptensor,2,feature)
#    h_sct3 = SCT1st(sptensor,3,feature)

    h_sct1,h_sct2,h_sct3,h_sct4 = SCT1stv2(sptensor,4,feature)

    return h_sct1,h_sct2,h_sct3,h_sct4

def SCT1stv2(sptensor,order,feature):
    '''
    sptensor = W
   '''
    degrees = torch.sum(sptensor,0)
    D = degrees
#    D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=1)
    iteration = 2**order
    scale_list = list(2**i - 1 for i in range(order+1))
#    scale_list = [0,1,3,7]
    feature_p = feature
    sct_diffusion_list = []
    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.matmul(sptensor,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
        if i in scale_list:
            sct_diffusion_list += [feature_p,]
    sct_feature1 = sct_diffusion_list[0]-sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1]-sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2]-sct_diffusion_list[3]
    sct_feature4 = sct_diffusion_list[3]-sct_diffusion_list[4]
    return sct_feature1,sct_feature2,sct_feature3,sct_feature4


