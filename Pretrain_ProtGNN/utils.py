import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from Pretrain_ProtGNN.structgen import protein_features 
except:
    from structgen import protein_features
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_parallel(module, input, device_ids=["cuda:0"], output_device="cpu"):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

def read_json(path):
    with open(path, 'r') as f:
        X = json.load(f)
    return X  

def ProteinFeatures(top_k=30, num_rbf=16, features_type='full', direction='bidirectional'):
    features = protein_features.ProteinFeatures(
            top_k=top_k, num_rbf=num_rbf,
            features_type=features_type,
            direction=direction
    )
    return features

alphabet = '#ACDEFGHIKLMNPQRSTVWY'
def completize(batch, batch_idx=1):
    """
    Note: For a batch of size 1
    """
    L_max = len(batch['seq'])
    X = np.zeros([batch_idx, L_max, 4, 3])
#     S = np.zeros([batch_idx, L_max], dtype=np.int32)
    S = batch['seq']
    mask = np.zeros([batch_idx, L_max], dtype=np.float32)

    # Build the batch
    x = np.stack([batch['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
    X[batch_idx-1,:len(x),:,:] = x
    
    l = len(batch['seq'])
#     indices = np.asarray([alphabet.index(a) for a in batch['seq']], dtype=np.int32)
#     S[batch_idx-1, :l] = indices
    mask[batch_idx-1, :l] = 1.

    # Remove NaN coords
    mask = mask * np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    isnan = np.isnan(X)
    X[isnan] = 0.

    # Conversion
#     S = torch.from_numpy(S).long().cuda()
    X = torch.from_numpy(X).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()
    return X, S, mask
  
def to_torch_geom(E, E_idx):
    # get edge indices
    node1 = torch.tensor([], dtype=torch.long)
    for idx in range(E_idx.size(-2)):
        node1 = torch.cat((node1, torch.full((1, E_idx.size(-1)), idx)), axis=1)
    node1 = node1.repeat((E_idx.size(0), 1))
    node2 = E_idx.view(E_idx.size(0), E_idx.size(1)*E_idx.size(2))
    E_idx = torch.stack((node1, node2), 1)

    # get edge attr
    E = E.view(E.size(0), E.size(1)*E.size(2), E.size(3))
   
    return E, E_idx.type(torch.LongTensor)
  
def prepare_data(path):
    X = read_json(path)
    features = ProteinFeatures()
    data_lst = []
    for i, x in enumerate(X):
        hchain = completize(x) 
        V, E, E_idx = features(hchain[0], hchain[-1]) 
        V, E, E_idx = V.to('cpu'), E.to('cpu'), E_idx.to('cpu') 
        E, E_idx = to_torch_geom(E, E_idx)
        data_lst.append(Data(x=V[0, :, :], edge_index=E_idx[0, :, :], edge_attr=E[0, :, :], batch=torch.tensor([int(i)])))
    return data_lst
        
def prepare_dataABAG(pathAB, pathAG):
    XAB = read_json(pathAB)
    XAB = XAB
    XAG = read_json(pathAG)
    XAG = XAG
    features = ProteinFeatures()
    data_lstAB, data_lstAG, sample_idx = [], [], 0
    print('Preparing Data...')
    for i, (xAB, xAG) in enumerate(tqdm(zip(XAB, XAG))):
        # try:
        hchainAB, hchainAG = completize(xAB), completize(xAG)

        V_AB, E_AB, E_idx_AB = features(hchainAB[0], hchainAB[-1]) 
        V_AB, E_AB, E_idx_AB = V_AB.to('cpu'), E_AB.to('cpu'), E_idx_AB.to('cpu') 
        E_AB, E_idx_AB = to_torch_geom(E_AB, E_idx_AB)
        seqAB_1 = hchainAB[1]

        V_AG, E_AG, E_idx_AG = features(hchainAG[0], hchainAG[-1]) 
        V_AG, E_AG, E_idx_AG = V_AG.to('cpu'), E_AG.to('cpu'), E_idx_AG.to('cpu') 
        E_AG, E_idx_AG = to_torch_geom(E_AG, E_idx_AG)
        seqAG_1 = hchainAG[1]
        
        dataAB_1 = Data(x=V_AB[0, :, :], edge_index=E_idx_AB[0, :, :],\
                        edge_attr=E_AB[0, :, :], batch=torch.tensor([sample_idx]),\
                        y=torch.tensor([1]))
        dataAG_1 = Data(x=V_AG[0, :, :], edge_index=E_idx_AG[0, :, :],\
                        edge_attr=E_AG[0, :, :], batch=torch.tensor([sample_idx]),\
                        y=torch.tensor([1]))

        sample_idx += 1

        # Getting Negative Samples 
        idx = random.choices([j for j in range(len(XAG)) if j!=i], k=1)[0]

        hchainAG = completize(XAG[idx])

        V_AG, E_AG, E_idx_AG = features(hchainAG[0], hchainAG[-1]) 
        V_AG, E_AG, E_idx_AG = V_AG.to('cpu'), E_AG.to('cpu'), E_idx_AG.to('cpu') 
        E_AG, E_idx_AG = to_torch_geom(E_AG, E_idx_AG)
        seqAG_2 = hchainAG[1]

        dataAB_2 = Data(x=V_AB[0, :, :], edge_index=E_idx_AB[0, :, :],\
                        edge_attr=E_AB[0, :, :], batch=torch.tensor([sample_idx]),\
                        y=torch.tensor([0]))
        dataAG_2 = Data(x=V_AG[0, :, :], edge_index=E_idx_AG[0, :, :],\
                        edge_attr=E_AG[0, :, :], batch=torch.tensor([sample_idx]),\
                        y=torch.tensor([0]))

        # Appending 
        data_lstAB.append((dataAB_1, seqAB_1, torch.tensor([1])))
        data_lstAB.append((dataAB_2, seqAB_1, torch.tensor([0])))
        data_lstAG.append((dataAG_1, seqAG_1, torch.tensor([1])))
        data_lstAG.append((dataAG_2, seqAG_2, torch.tensor([0])))
        sample_idx += 1
        # except:pass
    return data_lstAB, data_lstAG

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

def accuracy(sim, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    mask = torch.eye(sim.shape[0], dtype=torch.bool)
    neg = sim[~mask].view(sim.shape[0], -1)
    pos = sim[mask].view(sim.shape[0], -1)
    output = torch.cat([pos, neg], dim=1)
    target = torch.zeros(output.shape[0], dtype=torch.long)
    
    with torch.no_grad():
        batch_size = target.size(0)
        maxk = min(max(topk), batch_size)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        assert correct.size(1) == batch_size
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
        return res

def metric(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    '''
    
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 
    
    y_pred = (y_pred > 0.5).type(torch.uint8)  
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    
    result_df = pd.DataFrame([[precision.item(), recall.item(), f1.item()]], columns=['Precision', 'Recall', 'F1 Score'])
    return result_df
