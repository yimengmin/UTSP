import numpy as np
import torch

def TSPLoss(SctOutput,distance_matrix,num_of_nodes,device = 'cuda'):
    '''
    input:
    SctOutput: batchsize * num_of_nodes * num_of_nodes tensor
    distance_matrix: batchsize * num_of_nodes * num_of_nodes tensor
    '''
    HeatM = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 1, 2),-1, 1))
    weighted_path = torch.mul(HeatM, distance_matrix)
    weighted_path = weighted_path.sum(dim=(1,2))
    return weighted_path, HeatM

def get_heat_map(SctOutput,num_of_nodes,device = 'cuda'):
    '''
    input:
    SctOutput: batchsize * num_of_nodes * num_of_nodes tensor
    '''
    HeatM = torch.matmul(SctOutput, torch.roll(torch.transpose(SctOutput, 1, 2),-1, 1))
    return HeatM


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


