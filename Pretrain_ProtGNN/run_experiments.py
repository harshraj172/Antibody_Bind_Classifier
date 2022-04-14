import argparse
import numpy as np
import wandb

import torch
import os.path as osp
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from itertools import cycle
from torch_geometric.data import Data, DataLoader

from models.contrast_model import *
from models.gnn import *
from loss import *
from utils import *
import utils


def train(encoder_model, contrast_model, dataloader, optimizer, topk=1):
    encoder_model.train()
    epoch_loss, epoch_acc, num_batches = 0, 0, 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss, acc = contrast_model(g1=g1, g2=g2, batch=data.batch, topk=topk)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if data.num_graphs > 1:
            epoch_acc += acc
            num_batches += 1
    return epoch_loss, epoch_acc/num_batches


def test(encoder_model, path_Ab, path_Ag, batch_size):
    encoder_model.eval()
    data_lst_Ab = prepare_data(path_Ab, batch_size)
    data_lst_Ag = prepare_data(path_Ag, batch_size)
    loader1 = DataLoader(data_lst_Ab, batch_size=batch_size)
    loader2 = DataLoader(data_lst_Ag, batch_size=batch_size)
    
    acc, num_samples = 0, 0
    
    for i, (data1, data2) in enumerate(zip(cycle(loader1), loader2)):
        data1 = data1.to(args.device)
        data2 = data2.to(args.device)
        _, g1, _, _, _, _ = encoder_model(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        _, g2, _, _, _, _ = encoder_model(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        
        # Accuracy
        sim = utils._similarity(g1, g2)
        mask = torch.eye(sim.shape[0], dtype=torch.bool).to(args.device)
        neg = sim[~mask].view(sim.shape[0], -1)
        pos = sim[mask].view(sim.shape[0], -1)
        acc += torch.mean(pos.view(-1)) - torch.mean(neg.view(-1))
        
        if data1.num_graphs>1:
            num_samples += 1 

    return acc/num_samples


def main():
    data_lst = prepare_data(args.data_path, args.batch_size)
    dataloader = DataLoader(data_lst, batch_size=args.batch_size)
    node_dim = max(data_lst[0].x.size(-1), 1)
    edge_dim = max(data_lst[0].edge_attr.size(-1), 1)

    aug1 = A.Identity()
#     aug1 = A.PPRDiffusion(alpha=0.2, use_cache=False)
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=args.walk_length),
                           A.NodeDropping(pn=args.pn),
                           A.FeatureMasking(pf=args.pf),
                           A.EdgeRemoving(pe=args.pe)], 1)
    gconv = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(args.device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=args.temperature), mode='G2G').to(args.device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=args.epochs-1, desc='(T)') as pbar:
        for epoch in range(args.epochs):
            loss, acc = train(encoder_model, contrast_model, dataloader, optimizer, args.topk)
            if (epoch+1) % args.print_feq == 0:
                print(f"Top-{args.topk} Accuracy: {acc}")
                # ...log the running loss
                wandb.log({"train loss": loss})
                
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_acc = test(encoder_model, args.path_Ab, args.path_Ag, args.test_batch_size)
    print(f"Test Accuracy = {test_acc}")
    wandb.log({"test accuracy": test_acc})
#     print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Protein Pretraining')

    parser.add_argument('--device', default=device, type=str)
    parser.add_argument('--data_type', default="pdb", choices=["pdb", "swiss_prot"], type=str) 
    parser.add_argument('--data_path', default="pdb_data.json", type=str) 
    parser.add_argument('--walk_length', default=100, type=int)
    parser.add_argument('--pn', default=0.5, type=float, help="prob of dropping nodes")
    parser.add_argument('--pf', default=0.5, type=float, help="prob of masking features")
    parser.add_argument('--pe', default=0.5, type=float, help="prob of removing edges")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--num_layers', default=8, type=int)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--topk', default=(1,5), type=tuple)
    parser.add_argument('--print_feq', default=1, type=int, help="print the accuracy and log the running loss after certain interval")
    
    parser.add_argument('--path_Ab', default="data/SabDab/X_Ab.json", type=str)
    parser.add_argument('--path_Ag', default="data/SabDab/X_Ag.json", type=str)
    parser.add_argument('--test_batch_size', default=10, type=int)
    args = parser.parse_args()
    
    # find data length 
    X = read_json(args.data_path)
    
    config = {
      "dataset": args.data_type,
      "num samples": len(X),
      "walk length": args.walk_length,
      "pn": args.pn,
      "pf": args.pf,
      "pe": args.pe,
      "batch size": args.batch_size,
      "learning_rate": args.lr,
      "epochs": args.epochs,
      "hidden dim": args.hidden_dim,
      "num layers": args.num_layers,
      "temperature": args.temperature,  
    }
    
    del X
    
    wandb.init(config=config, project="Pretrain-GNN", entity="harsh1729")
    main()
