import re
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GINEConv
# from tape import ProteinBertModel, TAPETokenizer
from transformers import BertModel

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
                            nn.Linear(project_dim, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True))

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
        
    def forward(self, GAB, GAG):
        zAB, gAB = self.encoderAB(GAB.x, GAB.edge_index, GAB.edge_attr, GAB.batch)
        zAG, gAG = self.encoderAG(GAG.x, GAG.edge_index, GAG.edge_attr, GAG.batch)
        
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
                 pretrained_lm_dim=1024, project_dim=64):
        super(SeqEncoder, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_lm_model)
        self.project = torch.nn.Sequential(
                            nn.Linear(pretrained_lm_dim, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True))
    
    def forward(self, token_ids):
        output = self.model(token_ids)['pooler_output']
        h = self.project(output)
        return h
    
class StructSeqEnc(torch.nn.Module):
    def __init__(self, struct_enc, seq_enc,\
                 use_struct=True, use_seq=False):
        super(StructSeqEnc, self).__init__()
        self.struct_enc = struct_enc
        self.seq_enc = seq_enc
        self.use_struct = use_struct
        self.use_seq = use_seq
        
    def forward(self, GAB, GAG, tokenAB, tokenAG):
        h1, h2 = torch.zeros((tokenAB.size(0), 1)).cuda(), torch.zeros((tokenAG.size(0), 1)).cuda()
        if self.use_struct:
            zAB, zAG, gAB, gAG = self.struct_enc(GAB, GAG)
            gAB = self.struct_enc.encoderAB.project(gAB)
            gAG = self.struct_enc.encoderAG.project(gAG)
            h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)        
        if self.use_seq:
            sAB, sAG = self.seq_enc(tokenAB), self.seq_enc(tokenAG)
            h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)
        
        h = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)
        return h
