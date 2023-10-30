import torch
import numpy as np

def GCN_diffusion(W,order,feature,device='cuda'):
    """
    W: [batchsize,n,n]
    feature: [batchsize,n,n]
    """
    identity_matrices = torch.eye(W.size(1)).repeat(W.size(0), 1, 1)
    I_n = identity_matrices.to(device)
    A_gcn = W + I_n #[b,n,n]
    ###
    degrees = torch.sum(A_gcn,2)
    degrees = degrees.unsqueeze(dim=2) # [b,n,1]
    D = degrees
    ##
    D = torch.pow(D, -0.5)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    for i in range(order):
        A_gcn_feature = D*A_gcn_feature
        A_gcn_feature = torch.matmul(A_gcn,A_gcn_feature) # batched matrix x batched matrix https://pytorch.org/docs/stable/generated/torch.matmul.html
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list

def scattering_diffusionS4(sptensor,feature):
    '''
    A_tilte,adj_p,shape(N,N)
    feature:shape(N,3) :torch.FloatTensor
    all on cuda
    '''
    h_sct1,h_sct2,h_sct3,h_sct4 = SCT1stv2(sptensor,4,feature)

    return h_sct1,h_sct2,h_sct3,h_sct4

def SCT1stv2(W,order,feature):
    '''
    W = [b,n,n]
    '''
    degrees = torch.sum(W,2)
    D = degrees
#    D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=2)
    iteration = 2**order
    scale_list = list(2**i - 1 for i in range(order+1))
#    scale_list = [0,1,3,7]
    feature_p = feature
    sct_diffusion_list = []
    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.matmul(W,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
        if i in scale_list:
            sct_diffusion_list += [feature_p,]
    sct_feature1 = sct_diffusion_list[0]-sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1]-sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2]-sct_diffusion_list[3]
    sct_feature4 = sct_diffusion_list[3]-sct_diffusion_list[4]
    return sct_feature1,sct_feature2,sct_feature3,sct_feature4


