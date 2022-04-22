import sys
import re 

import numpy as np
import torch

import esm
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

from transformers import BertModel, BertTokenizer


class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, node_feature_size: int, edge_dim: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, node_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, node_feature_size: int, edge_dim: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, node_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 64)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h


# class StructNet(nn.Module):
#     """SE(3) equivariant GCN with attention"""
#     def __init__(self, num_layers: int, node_feature_size: int, edge_dim: int, 
#                  num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
#                  div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
#         super().__init__()
#         # Build the network
#         self.num_layers = num_layers
#         self.num_nlayers = num_nlayers
#         self.num_channels = num_channels
#         self.num_degrees = num_degrees
#         self.edge_dim = edge_dim
#         self.div = div
#         self.pooling = pooling
#         self.n_heads = n_heads

#         self.fibers = {'in': Fiber(1, node_feature_size),
#                        'mid': Fiber(num_degrees, self.num_channels),
#                        'out': Fiber(1, num_degrees*self.num_channels)}

#         blocks_AB = self._build_gcn(self.fibers)
#         blocks_AG = self._build_gcn(self.fibers)
        
#         self.Gblock_AB, self.FCblock_AB = blocks_AB
#         self.Gblock_AG, self.FCblock_AG = blocks_AG
        
        
#     def _build_gcn(self, fibers):
#         # Equivariant layers
#         Gblock = []
#         fin = fibers['in']
#         for i in range(self.num_layers):
#             Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
#                                   div=self.div, n_heads=self.n_heads))
#             Gblock.append(GNormSE3(fibers['mid']))
#             fin = fibers['mid']
#         Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

#         # Pooling
#         if self.pooling == 'avg':
#             Gblock.append(GAvgPooling())
#         elif self.pooling == 'max':
#             Gblock.append(GMaxPooling())

#         # FC layers
#         FCblock = []
#         FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
#         FCblock.append(nn.ReLU(inplace=True))

#         return nn.ModuleList(Gblock), nn.ModuleList(FCblock)


#     def forward(self, G_AB, G_AG):
#         # Compute equivariant weight basis from relative positions
#         (basis_AB, r_AB), (basis_AG, r_AG) = get_basis_and_r(G_AB, self.num_degrees-1),\
#                                              get_basis_and_r(G_AG, self.num_degrees-1)

#         # encoder (equivariant layers)
#         h_AB, h_AG = {'0': G_AB.ndata['f']}, {'0': G_AG.ndata['f']}
#         # print(f"Antibody = {h_AB}")
#         # print(f"Antigen = {h_AG}")
#         for i in range(len(self.Gblock_AB)):
#             h_AB, h_AG = self.Gblock_AB[i](h_AB, G=G_AB, r=r_AB, basis=basis_AB),\
#                          self.Gblock_AG[i](h_AG, G=G_AG, r=r_AG, basis=basis_AG)
#         # print(f"Antibody = {h_AB}")
#         # print(f"Antigen = {h_AG}")
#         for i in range(len(self.FCblock_AB)):
#             h_AB, h_AG = self.FCblock_AB[i](h_AB), self.FCblock_AG[i](h_AG)
#         # print(f"Antibody = {h_AB}")
#         # print(f"Antigen = {h_AG}")
#         h = torch.diag(torch.matmul(h_AB, h_AG.permute(1, 0))).unsqueeze(-1)
#         # print(f"END = {h}")
#         return h

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
def tokenize(seqs, tokenizer=tokenizer, MAX_LEN=1024):
    seqs = [" ".join(seq) for seq in seqs]
    seqs = [re.sub(r"[UZOB]", "X", seq) for seq in seqs]
    token_ids = torch.cat([tokenizer(seq, max_length=MAX_LEN,\
                            truncation=True, padding='max_length',\
                            return_tensors='pt')['input_ids'] for seq in seqs])
    return token_ids

class SeqEncoder(torch.nn.Module):
    def __init__(self, pretrained_lm_model="Rostlab/prot_bert",\
                 pretrained_lm_dim=1024):
        super(SeqEncoder, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_lm_model)
        self.project = torch.nn.Sequential(
                            nn.Linear(pretrained_lm_dim, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True))
    
    def forward(self, token_ids):
        output = self.model(token_ids)['pooler_output']
        h = self.project(output)
        return h
    
class StructSeqEnc(nn.Module):
    def __init__(self, use_struct: bool, use_seq: bool, num_layers: int, node_feature_size: int, edge_dim: int,
                 num_channels: int=16, num_nlayers: int=1, num_degrees: int=4, div: int=4, pooling: str='avg',
                 n_heads: int=1, pretrained_lm_model: str="Rostlab/prot_bert", pretrained_lm_dim: int=1024, **kwargs):
        super().__init__()
        self.use_struct, self.use_seq = use_struct, use_seq
        
        if use_struct:
            self.Struct_EncoderAB = SE3Transformer(num_layers, 
                                                   node_feature_size, 
                                                   edge_dim,
                                                   num_channels=num_channels,
                                                   num_nlayers=num_nlayers,
                                                   num_degrees=num_degrees,
                                                   div=div,
                                                   pooling=pooling,
                                                   n_heads=n_heads)
            
            self.Struct_EncoderAG = SE3Transformer(num_layers, 
                                                   node_feature_size, 
                                                   edge_dim,
                                                   num_channels=num_channels,
                                                   num_nlayers=num_nlayers,
                                                   num_degrees=num_degrees,
                                                   div=div,
                                                   pooling=pooling,
                                                   n_heads=n_heads)
        if use_seq:
            self.Seq_EncoderAB = SeqEncoder(pretrained_lm_model,
                                          pretrained_lm_dim)
            self.Seq_EncoderAG = SeqEncoder(pretrained_lm_model,
                                          pretrained_lm_dim)
    
    def forward(self, gAB, tokenAB, gAG, tokenAG):
        h1, h2 = torch.zeros(tokenAB.size(0), 1).cuda(), torch.zeros(tokenAG.size(0), 1).cuda()
                                
        if self.use_struct:
            gAB, gAG = self.Struct_EncoderAB(gAB), self.Struct_EncoderAG(gAG)
            h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)
        if self.use_seq:
            sAB, sAG = self.Seq_EncoderAB(tokenAB), self.Seq_EncoderAG(tokenAG)
            h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)
        
        h = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)     
        h = torch.sigmoid(h)
        return h
