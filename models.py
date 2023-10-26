import torch
import torch.nn.functional as F
from torch import tensor
import torch.nn
from diff_moduleS4p import scattering_diffusionS4,GCN_diffusion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
class SCTConv(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))

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
        B = support0.size(0) # batchsize
        N = support0.size(1) # n
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
        a_input_A = torch.cat((h, h_A), dim=2).unsqueeze(1)
        a_input_A2 = torch.cat((h, h_A2), dim=2).unsqueeze(1)
        a_input_A3 = torch.cat((h, h_A3), dim=2).unsqueeze(1)
        a_input_sct1 = torch.cat((h, h_sct1), dim=2).unsqueeze(1)
        a_input_sct2 = torch.cat((h, h_sct2), dim=2).unsqueeze(1)
        a_input_sct3 = torch.cat((h, h_sct3), dim=2).unsqueeze(1)
        a_input_sct4 = torch.cat((h, h_sct4), dim=2).unsqueeze(1) # [b,1,n,2f]
# S3
#        a_input = torch.cat((a_input_A, a_input_A2, a_input_A3, a_input_sct1, a_input_sct2, a_input_sct3), 1).view(N, 6, -1)
# S4
#        a_input = torch.cat((a_input_A, a_input_A2, a_input_A3, a_input_sct1, a_input_sct2, a_input_sct3,a_input_sct4), 1).view(N, 7, -1)
# 1 low pass + 3 high pass
        a_input = torch.cat((a_input_A, a_input_A2,a_input_sct1, a_input_sct2, a_input_sct3,a_input_sct4), 1).view(B,6,N,-1)
#        print('a_input shape')
#        print(a_input.size())
        # xxxx stop

        e = torch.matmul(nn.functional.relu(a_input), self.a).squeeze(3)
#        print('e shape')
#        print(e.size())
# S3
#        attention = F.softmax(e, dim=1).view(N, 6, -1)
# S4
        attention = F.softmax(e, dim=1).view(B,6, N, -1)
#        print('attention size')
#        print(attention.size())
# S3
#        h_all = torch.cat((h_A.unsqueeze(dim=2), h_A2.unsqueeze(dim=2), h_A3.unsqueeze(dim=2),
#        h_sct1.unsqueeze(dim=2), h_sct2.unsqueeze(dim=2), h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        # h_A: [b,n,f]
# S4
        h_all = torch.cat((h_A.unsqueeze(dim=1), h_A2.unsqueeze(dim=1),h_sct1.unsqueeze(dim=1), h_sct2.unsqueeze(dim=1), h_sct3.unsqueeze(dim=1), h_sct4.unsqueeze(dim=1)),dim=1).view(B, 6,N,-1)
        h_prime = torch.mul(attention, h_all)
        h_prime = torch.mean(h_prime, 1) # (B,n,f)
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X

class GNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim))

        self.mlp1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.m = nn.Softmax(dim=1)
#        self.mlp1 = Linear(hidden_dim * (1 + n_layers), output_dim)

    def forward(self, X, adj, moment=1, device='cuda'):
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X, adj, moment=moment, device=device)
            hidden_states = torch.cat([hidden_states, X], dim=-1)

        X = hidden_states
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X)
        X = self.m(X)
        return X


