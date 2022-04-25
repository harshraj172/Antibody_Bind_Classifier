import json
import random
import numpy as np
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from Pretrain_ProtGNN.structgen import protein_features 
import dgl

from Pretrain_ProtGNN.utils import *

DTYPE = np.float32
class Antibody_Antigen_Dataset(Dataset):
    node_feature_size, edge_feature_size = 6, 39
    """Class for Antibody and Antigen data"""
    def __init__(self, path_X_Ab, path_X_Ag, transform=None):
        self.X_Ab = read_json(path_X_Ab)
        self.X_Ag = read_json(path_X_Ag)
        
        self.data_list = self.create_data()

    def get_structseq_lst(self, X):
        print('Preparing Data')
        features = ProteinFeatures()
        
        data_lst = []
        for x in tqdm(X):
            hchain = completize(x)
            if hchain[0].size()[-1] == 3 and\
               hchain[0].size()[-2] == 4:
                x_ca = np.array(hchain[0][0, :, 1, :].cpu())
                V, E, E_idx = features(hchain[0], hchain[-1])
                V, E, E_idx = V.cpu(), E.cpu(), E_idx.cpu()
                E, E_idx = to_torch_geom(E, E_idx)

                ## dgl graph
                # adjacency = nx.adjacency_matrix(nxGraph)
                adjacency = to_dense_adj(E_idx[0, :, :])[0, :, :]
                adjacency = np.array(adjacency)
                src, dst = np.nonzero(adjacency)
                dglGraph = dgl.graph((src, dst))

                # add node features
                dglGraph.ndata['x'] = torch.tensor(x_ca)
                dglGraph.ndata['f'] = V[0, :, :].unsqueeze(-1)

                # add edge features
                dglGraph.edata['d'] = torch.tensor(x_ca[dst] - x_ca[src])
                dglGraph.edata['w'] = E[0, :, :]

                data_lst.append((dglGraph, hchain[1]))

        return data_lst

    def create_data(self):
        structseq_list_Ab = self.get_structseq_lst(self.X_Ab)
        structseq_list_Ag = self.get_structseq_lst(self.X_Ag)
        data_lst = [{'Antibody': {'struct': struct_Ab, 'seq': seq_Ab}, 
                     'Antigen': {'struct': struct_Ag, 'seq': seq_Ag}, 
                     'target': np.asarray([1], dtype=DTYPE)} 
                    for ((struct_Ab, seq_Ab), (struct_Ag, seq_Ag)) in zip(structseq_list_Ab, structseq_list_Ag)]

        print('Getting Negative Samples')
        for i in tqdm(range(len(structseq_list_Ab))):
            tmp_structseq_list_Ag = structseq_list_Ag.copy()
            tmp_structseq_list_Ag.pop(i)

            (struct_Ag, seq_Ag) = random.choices(tmp_structseq_list_Ag, k=1)[0]
            
            data_lst.append({'Antibody': {'struct': structseq_list_Ab[i][0], 'seq': structseq_list_Ab[i][1]}, 
                            'Antigen': {'struct': struct_Ag, 'seq': seq_Ag}, 
                            'target': np.asarray([0], dtype=DTYPE)})

        random.shuffle(data_lst)
        return data_lst
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]["Antibody"], self.data_list[idx]["Antigen"], self.data_list[idx]["target"] 
