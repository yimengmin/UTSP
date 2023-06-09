import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
import networkx as nx
import scipy
from utils import TSPLoss,edge_overlap
import pickle
from torch.utils.data import  Dataset,DataLoader# use pytorch dataloader
from random import shuffle
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=100, help='Graph Size')
parser.add_argument('--dropout', type=float, default=0.0,help='probability of an element to be zeroed:')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=2,
                    help='num of layers')
parser.add_argument('--use_smoo', action='store_true')
parser.add_argument('--EPOCHS', type=int, default=100,
                    help='epochs to train')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=2.,
                    help='temperature for adj matrix')
parser.add_argument('--rescale', type=float, default=1.,
                    help='rescale for xy plane')
parser.add_argument('--C1_penalty', type=float, default=20.,
                    help='penalty row/column')
parser.add_argument('--topk', type=int, default=10,
                    help='topk')
parser.add_argument('--diag_loss', type=float, default=0.1,
                    help='penalty on the diag')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
### load train instance

tsp_instances = np.load('data/train_tsp_instance_%d.npy'%args.num_of_nodes)
NumofTestSample = tsp_instances.shape[0]

Std = np.std(tsp_instances, axis=1)
Mean = np.mean(tsp_instances, axis=1)


tsp_instances = tsp_instances - Mean.reshape((NumofTestSample,1,2))
tsp_instances = args.rescale * tsp_instances # 2.0 is the rescale
tsp_sols = np.load('data/train_tsp_sol_%d.npy'%args.num_of_nodes)



import json
total_samples = tsp_instances.shape[0]

#preposs_time = time.time()

from models import GNN,GCN
#scattering model
model = GNN(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers,dropout=args.dropout,Withgres=args.use_smoo,smooth=args.smoo)
from scipy.spatial import distance_matrix

### count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total number of parameters:')
print(count_parameters(model))


#dis_mat = distance_matrix(tsp_instances[0],tsp_instances[0])
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html
def coord_to_adj(coord_arr):
    dis_mat = distance_matrix(coord_arr,coord_arr)
#    return np.exp(-(1./temperature)*dis_mat)
#    dis_mat = dis_mat + np.eye(args.num_of_nodes)*args.diag_penalty # add a large number on the diagonal,can be intrepreted as a constrint
    return dis_mat

tsp_instances_adj = np.zeros((total_samples,args.num_of_nodes,args.num_of_nodes))
for i in range(total_samples):
    tsp_instances_adj[i] = coord_to_adj(tsp_instances[i])
#print(coord_to_adj(tsp_instances[0]))
class TSP_Dataset(Dataset):
    def __init__(self, coord,data, targets):
        self.coord = torch.FloatTensor(coord)
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        xy_pos = self.coord[index]
        x = self.data[index]
        y = self.targets[index]
#        tsp_instance = Data(coord=x,sol=y)
        return tuple(zip(xy_pos,x,y))

    def __len__(self):
        return len(self.data)

dataset = TSP_Dataset(tsp_instances,tsp_instances_adj,tsp_sols)
num_trainpoints = total_samples - 1000 # training on 2,000 intances, you can improve this size
print('num_trainpoints: %d'%num_trainpoints)
num_valpoints = total_samples - num_trainpoints
sctdataset = dataset
traindata= sctdataset[0:num_trainpoints]
valdata = sctdataset[num_trainpoints:]
batch_size = args.batch_size
train_loader = DataLoader(traindata, batch_size, shuffle=True)
val_loader =  DataLoader(valdata, batch_size, shuffle=False)



from torch.optim.lr_scheduler import StepLR
#optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.wdecay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wdecay)
scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
def train(epoch):
    scheduler.step()
    model.cuda()
    model.train()
    print('Epoch: %d'%epoch)
    for batch in train_loader:
        batchloss = 0.0
        for i in range(len(batch[0])):
            xy_pos = batch[0][i]
            distance_m = batch[1][i] #distance_m is used to calculate the loss function
            sol = batch[2][i] # sol is not used during UL training
            distance_m = distance_m.cuda()
            adj = torch.exp(-1.*distance_m/args.temperature)
            adj.fill_diagonal_(0)
            adj = adj.cuda()
            features = xy_pos.cuda()
            output = model(features,adj,moment = args.moment)
            TSPLoss_constaint,Heat_mat = TSPLoss(SctOutput=output,distance_matrix=distance_m,num_of_nodes=args.num_of_nodes)
            Diag_loss =  torch.diagonal(Heat_mat, 0)
            Nrmlzd_constraint2 = (1. - torch.sum(output,1))**2
            Nrmlzd_constraint2 = torch.sum(Nrmlzd_constraint2)
            loss = args.C1_penalty*Nrmlzd_constraint2 + 1.*TSPLoss_constaint + args.diag_loss*torch.sum(Diag_loss)
            batchloss += loss
        batchloss = batchloss/len(batch[0])
        print('Loss: %.5f'%batchloss.item())
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
import os
save_dir_path = 'Saved_Models/TSP_%d/'%(args.num_of_nodes)
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
for i in range(args.EPOCHS):
    train(i)
    torch.save(model.state_dict(),save_dir_path+'scatgnn_layer_%d_hid_%d_model_%d_temp_%.3f.pth'%(args.nlayers,args.hidden,i,args.temperature))

