import re
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GINEConv
# from tape import ProteinBertModel, TAPETokenizer
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class AugEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(AugEncoder, self).__init__()
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

class DualEncoder(torch.nn.Module):
    def __init__(self, encoderAB, encoderAG):
        super(DualEncoder, self).__init__()
        self.encoderAB = encoderAB
        self.encoderAG = encoderAG
        
    def forward(self, xAB, edge_indexAB, edge_attrAB, batchAB,\
                xAG, edge_indexAG, edge_attrAG, batchAG):
        zAB, gAB = self.encoderAB(xAB, edge_indexAB, edge_attrAB, batchAB)
        zAG, gAG = self.encoderAG(xAG, edge_indexAG, edge_attrAG, batchAG)
        
        return zAB, zAG, gAB, gAG
    
def tokenize(seqs, tokenizer, MAX_LEN=1024):
    seqs = [" ".join(seq) for seq in seqs]
    seqs = [re.sub(r"[UZOB]", "X", seq) for seq in seqs]
    token_ids = torch.cat([tokenizer(seq, max_length=MAX_LEN,\
                            truncation=True, padding='max_length',\
                            return_tensors='pt')['input_ids'] for seq in seqs])
    return token_ids

class SeqEncoder(torch.nn.Module):
    def __init__(self, pretrained_lm_model="Rostlab/prot_bert",\
                 pretrained_lm_dim=526, project_dim=64):
        super(SeqEncoder, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_lm_model)
        self.project = torch.nn.Sequential(
            nn.Linear(pretrained_lm_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
    
    def forward(self, token_ids):
        output = self.model(token_ids)['pooler_output']
        h = self.project(output)
        return h 
