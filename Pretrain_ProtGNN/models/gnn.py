import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GINEConv

def make_gine_conv(node_dim, edge_dim, out_dim):
    return GINEConv(nn=nn.Sequential(nn.Linear(node_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)), edge_dim=edge_dim)


class GConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gine_conv(node_dim, edge_dim, hidden_dim))
            else:
                self.layers.append(make_gine_conv(hidden_dim, edge_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index, edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_attr, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_attr)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_attr)
        z, g = self.encoder(x, edge_index, edge_attr, batch)
        z1, g1 = self.encoder(x1, edge_index1, edge_weight1, batch)
        z2, g2 = self.encoder(x2, edge_index2, edge_weight2, batch)
        return z, g, z1, z2, g1, g2
