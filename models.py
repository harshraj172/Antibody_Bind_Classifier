import sys

import numpy as np
import torch

import esm
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
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
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
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

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
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


class StructNet(nn.Module):
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

        blocks_AB = self._build_gcn(self.fibers)
        blocks_AG = self._build_gcn(self.fibers)
        
        self.Gblock_AB, self.FCblock_AB = blocks_AB
        self.Gblock_AG, self.FCblock_AG = blocks_AG
        
        
    def _build_gcn(self, fibers):
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

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)


    def forward(self, G_AB, G_AG):
        # Compute equivariant weight basis from relative positions
        (basis_AB, r_AB), (basis_AG, r_AG) = get_basis_and_r(G_AB, self.num_degrees-1),\
                                             get_basis_and_r(G_AG, self.num_degrees-1)

        # encoder (equivariant layers)
        h_AB, h_AG = {'0': G_AB.ndata['f']}, {'0': G_AG.ndata['f']}
        # print(f"Antibody = {h_AB}")
        # print(f"Antigen = {h_AG}")
        for i in range(len(self.Gblock_AB)):
            h_AB, h_AG = self.Gblock_AB[i](h_AB, G=G_AB, r=r_AB, basis=basis_AB),\
                         self.Gblock_AG[i](h_AG, G=G_AG, r=r_AG, basis=basis_AG)
        # print(f"Antibody = {h_AB}")
        # print(f"Antigen = {h_AG}")
        for i in range(len(self.FCblock_AB)):
            h_AB, h_AG = self.FCblock_AB[i](h_AB), self.FCblock_AG[i](h_AG)
        # print(f"Antibody = {h_AB}")
        # print(f"Antigen = {h_AG}")
        h = torch.diag(torch.matmul(h_AB, h_AG.permute(1, 0))).unsqueeze(-1)
        # print(f"END = {h}")
        return h


class get_SeqEmb:
    def __init__(self, pretrained_model_name):
        self.model, self.alphabet, self.batch_converter = self.load_pretrained(pretrained_model_name)
    def load_pretrained(self, pretrained_model_name):
        if pretrained_model_name=="esm1b_t33_650M_UR50S":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            batch_converter = alphabet.get_batch_converter()
            model = model.to(device)
        return model, alphabet, batch_converter
    
    def pretrained_emb(self, data):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        # Extract per-residue representations (on CPU)
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0).unsqueeze(0))
        return torch.cat(sequence_representations)
    
    
class StructSeqNet(nn.Module):
    def __init__(self, use_struct: bool, use_seq: bool, num_layers: int, node_feature_size: int, edge_dim: int,
                 pretrained_lm_model: str='esm1b_t33_650M_UR50S', pretrained_lm_emb_dim: int=1280, num_channels: int=16, num_nlayers: int=1, 
                 num_degrees: int=4, div: int=4, pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        self.use_struct, self.use_seq = use_struct, use_seq
        
        self.StructModel = StructNet(num_layers, 
                                   node_feature_size, 
                                   edge_dim,
                                   num_channels=num_channels,
                                   num_nlayers=num_nlayers,
                                   num_degrees=num_degrees,
                                   div=div,
                                   pooling=pooling,
                                   n_heads=n_heads).to(device)
        self.comblayers = nn.ModuleList([nn.Linear(2, 1),
                                     nn.ReLU(inplace=True)])
        
        self.FCBlock_AB, self.FCBlock_AG = self.FCBlock(pretrained_lm_emb_dim), self.FCBlock(pretrained_lm_emb_dim)
        
    def FCBlock(self, in_dim):
        layers = nn.ModuleList([nn.Linear(in_dim, int(in_dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_dim/2), int(in_dim/4)),
                                nn.ReLU(inplace=True)])
        return layers 
    
    def forward(self, gAB, hAB, gAG, hAG):
        h1, h2 = torch.zeros(hAB.size(0), 1).to(device), torch.zeros(hAG.size(0), 1).to(device)
                                
        if self.use_struct:
            h1 = self.StructModel(gAB, gAG)
        if self.use_seq:
            for i in range(len(self.FCBlock_AB)):
                hAB, hAG = self.FCBlock_AB[i](hAB), self.FCBlock_AG[i](hAG)

            # Dot product of hidden states
            h2 = torch.diag(torch.matmul(hAB, hAG.permute(1, 0))).unsqueeze(-1)
        
        h = torch.cat((h1, h2), dim=1)
                                
        for layer in self.comblayers:
            print(h)
            h = layer(h)
                                
        return h
