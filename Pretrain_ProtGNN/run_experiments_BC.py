import random
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.contrast_model import *
from models.gnn import *
from loss import *
from utils import *
import utils

from transformers import BertTokenizer

def to_np(x):
    return x.cpu().detach().numpy()

def train(encoder_model, seq_encoder, tokenizer, dataloaderAB, dataloaderAG, optimizer):
    if args.use_struct:
        encoder_model.train()
    if args.use_seq:
        seq_encoder.train()
    epoch_loss = 0
    for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
        (GAB, seqAB, y) = dataAB
        (GAG, seqAG, y) = dataAG
        GAB = GAB.to(args.device)
        GAG = GAG.to(args.device)
        y = y.reshape(-1, 1).to(torch.float32).to(args.device)
        
        optimizer.zero_grad()
        
        h1, h2 = torch.zeros((y.size(0), 1)).to(args.device), torch.zeros((y.size(0), 1)).to(args.device)
        if args.use_struct:
            zAB, zAG, gAB, gAG = encoder_model(GAB.x, GAB.edge_index, GAB.edge_attr, GAB.batch,\
                                               GAG.x, GAG.edge_index, GAG.edge_attr, GAG.batch)
            gAB = encoder_model.encoderAB.project(gAB)
            gAG = encoder_model.encoderAG.project(gAG)
            h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)
        if args.use_seq:
            tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
            tokenAB, tokenAG = tokenAB.to(args.device), tokenAG.to(args.device) 
            sAB, sAG = seq_encoder(tokenAB), seq_encoder(tokenAG)
            h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)

        pred = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)
        loss, __ = BinaryClass_Loss(pred, y)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(dataloaderAB)


def test(encoder_model, seq_encoder, tokenizer, dataloaderAB, dataloaderAG):
    if args.use_struct:
        encoder_model.eval()
    if args.use_seq:
        seq_encoder.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
            (GAB, seqAB, y) = dataAB
            (GAG, seqAG, y) = dataAG
            GAB = GAB.to(args.device)
            GAG = GAG.to(args.device)
            y = y.reshape(-1, 1).to(torch.float32).to(args.device)

            h1, h2 = torch.zeros((y.size(0), 1)).to(args.device), torch.zeros((y.size(0), 1)).to(args.device)
            if args.use_struct:
                zAB, zAG, gAB, gAG = encoder_model(GAB.x, GAB.edge_index, GAB.edge_attr, GAB.batch,\
                                                   GAG.x, GAG.edge_index, GAG.edge_attr, GAG.batch)
                gAB = encoder_model.encoderAB.project(gAB)
                gAG = encoder_model.encoderAG.project(gAG)
                h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)
            if args.use_seq:
                tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
                tokenAB, tokenAG = tokenAB.to(args.device), tokenAG.to(args.device) 
                sAB, sAG = seq_encoder(tokenAB), seq_encoder(tokenAG)
                h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)

            pred = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)
            loss, __ = BinaryClass_Loss(pred, y)

            result_df = metric(pred.view(-1), y.long().view(-1))

            epoch_loss += loss.item()

    return epoch_loss/len(dataloaderAB), result_df


def main(
    
    ):
    train_data_lstAB, train_data_lstAG = prepare_dataABAG(f'{args.train_data_dir}/XAb.json', f'{args.train_data_dir}/XAg.json')
    train_loaderAB = DataLoader(train_data_lstAB[:200], batch_size=args.batch_size, shuffle=True)
    del train_data_lstAB
    
    node_dim = max(train_data_lstAG[0][0].x.size(-1), 1)
    edge_dim = max(train_data_lstAG[0][0].edge_attr.size(-1), 1)
    
    train_loaderAG = DataLoader(train_data_lstAG[:200], batch_size=args.batch_size, shuffle=True)
    del train_data_lstAG
  
    test_data_lstAB, test_data_lstAG = prepare_dataABAG(f'{args.test_data_dir}/XAb.json', f'{args.test_data_dir}/XAg.json')
    test_loaderAB = DataLoader(test_data_lstAB[-200:], batch_size=args.test_batch_size, shuffle=True)
    del test_data_lstAB
    test_loaderAG = DataLoader(test_data_lstAG[-200:], batch_size=args.test_batch_size, shuffle=True)
    del test_data_lstAG
    
    # Structure Model
    gconvAB = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    gconvAG = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    encoder_model = DualEncoder(encoderAB=gconvAB, encoderAG=gconvAG).to(args.device)
    
    # Sequence Model
    seq_encoder = SeqEncoder().to(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_lm_model, do_lower_case=False)
    
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=args.epochs-1, desc='(T)') as pbar:
        for epoch in range(args.epochs):
            loss = train(encoder_model, seq_encoder, tokenizer, train_loaderAB, train_loaderAG, optimizer)
            if (epoch+1) % args.print_feq == 0:
                # print(f"Top-{args.topk} Accuracy: {acc}")
                if args.log_wandb:
                    wandb.log({"Train Loss": loss})
                
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_loss, test_result_df = test(encoder_model, seq_encoder, tokenizer, test_loaderAB, test_loaderAG)
    print(f"Test Loss = {test_loss}")
    print(test_result_df)
    if args.log_wandb:
        wandb.log({"test L1 loss": test_loss,
                   "test precision": test_result_df['Precision'][0],
                   "test recall": test_result_df['Recall'][0], 
                   "test F1 score": test_result_df['F1 Score'][0]})


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Bind Classifer--Protein')

    parser.add_argument('--use_struct', default=True, type=eval)
    parser.add_argument('--use_seq', default=False, type=eval) 
    parser.add_argument('--pretrained_lm_model', default="Rostlab/prot_bert", type=str)
    parser.add_argument('--device', default=device, type=str)
    parser.add_argument('--data_type', default="pdb", choices=["pdb", "swiss_prot"], type=str) 
    parser.add_argument('--train_data_dir', default="data/SabDab/train", type=str)
    parser.add_argument('--test_data_dir', default="data/SabDab/train", type=str)    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=7, type=int)
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--topk', default=(1,5), type=tuple)
    parser.add_argument('--log_wandb', default=False, type=eval)
    parser.add_argument('--print_feq', default=1, type=int, help="print the accuracy after certain interval")
    
    args = parser.parse_args()
    
    # find data length 
    X = read_json(f'{args.train_data_dir}/XAb.json')
    
    config = {
      "dataset": args.data_type,
      "num samples": len(X),
      "batch size": args.batch_size,
      "learning_rate": args.lr,
      "epochs": args.epochs,
      "hidden dim": args.hidden_dim,
      "num layers": args.num_layers,
      "temperature": args.temperature,  
    }
    
    del X
    
    if args.log_wandb:
        wandb.init(config=config, project="Bind CLassifier--GINECONV", entity="harsh1729")
    main()
