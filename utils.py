'''
adapted from Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks
'''
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from time import perf_counter
from torch_geometric.utils.convert import to_scipy_sparse_matrix 
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def TSPLoss(SctOutput,distance_matrix,num_of_nodes,device = 'cuda'):
    '''
    input:
    SctOutput: num_of_nodes * num_of_nodes tensor
    distance_matrix: num_of_nodes * num_of_nodes tensor
    '''
    Tsp_point_wise_distance = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 0, 1),-1, 0))
    weighted_path = torch.mul(Tsp_point_wise_distance, distance_matrix)
    weighted_path = torch.sum(weighted_path)
    return weighted_path, Tsp_point_wise_distance

def get_heat_map(SctOutput,num_of_nodes,device = 'cuda'):
    '''
    input:
    SctOutput: num_of_nodes * num_of_nodes tensor
    '''
    Tsp_point_wise_distance = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 0, 1),-1, 0))
    return Tsp_point_wise_distance


def edge_overlap(pred,gt_sol):
    '''
    gt_sol: the ground truth solution: a list with num_of_nodes nodes
    pred: pred, an array with (num_of_nodes,top_k)
    '''
    gt_edge_set = set()
    for i in range(pred.shape[0]):
        gt_edge_set.add((int(gt_sol[i]),int(gt_sol[i+1])))
        gt_edge_set.add((int(gt_sol[i+1]),int(gt_sol[i])))
    pred_edge_set = set()
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_node = pred[i][j]
            pred_node = int(pred_node)
            if not i==pred_node:
                pred_edge_set.add((i,pred_node))
                pred_edge_set.add((pred_node,i))
    pred_gt_intsect = pred_edge_set.intersection(gt_edge_set)
    len_of_pred_gt = len(pred_edge_set)
    
    
    overlap_edge = len(pred_gt_intsect)/2 #here we consider bi-directional, so div 2
    return overlap_edge,len_of_pred_gt/2


