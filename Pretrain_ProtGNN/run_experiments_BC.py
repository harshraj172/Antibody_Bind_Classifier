import random
import argparse
import numpy as np
import wandb

import torch
import os.path as osp
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn, optim
from tqdm import tqdm
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


def train(encoder_model, seq_encoder, use_struct, use_seq, tokenizer,\
          dataloaderAB, dataloaderAG, optimizer1, optimizer2, device1, device2):
    if use_struct:
        encoder_model.train()
    if use_seq:
        seq_encoder.train()
    epoch_loss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
        (GAB, seqAB, y) = dataAB
        (GAG, seqAG, y) = dataAG
        y = y.reshape(-1, 1).to(torch.float32).to(device1)
        
        h1, h2 = torch.zeros((y.size(0), 1)).to(device1), torch.zeros((y.size(0), 1)).to(device2)
        if use_struct:
            GAB = GAB.to(device1)
            GAG = GAG.to(device1)
            optimizer1.zero_grad()
            
            zAB, zAG, gAB, gAG = encoder_model(GAB.x, GAB.edge_index, GAB.edge_attr, GAB.batch,\
                                               GAG.x, GAG.edge_index, GAG.edge_attr, GAG.batch)
            gAB = encoder_model.encoderAB.project(gAB)
            gAG = encoder_model.encoderAG.project(gAG)
            h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)
        if use_seq:
            optimizer2.zero_grad()
            
            tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
            tokenAB, tokenAG = tokenAB.to(device2), tokenAG.to(device2) 
            sAB, sAG = seq_encoder(tokenAB), seq_encoder(tokenAG)
            h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)
            
            h2.to(device1)

        pred = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)
        loss, __ = BinaryClass_Loss(pred, y)
        
        loss.backward()
        if use_struct:
            optimizer1.step()
        if use_seq:
            optimizer2.step()

        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
        
        epoch_loss += loss.item()
    
    result_df = metric(Y_pred.view(-1), Y_true.long().view(-1))
    return epoch_loss/len(dataloaderAB), result_df


def test(encoder_model, seq_encoder, use_struct, use_seq,\
         tokenizer, dataloaderAB, dataloaderAG, device1, device2):
    if use_struct:
        encoder_model.eval()
    if use_seq:
        seq_encoder.eval()
    
    with torch.no_grad():
        epoch_loss = 0
        Y_true, Y_pred = torch.tensor([]), torch.tensor([])
        for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
            (GAB, seqAB, y) = dataAB
            (GAG, seqAG, y) = dataAG
            y = y.reshape(-1, 1).to(torch.float32).to(device1)

            h1, h2 = torch.zeros((y.size(0), 1)).to(device1), torch.zeros((y.size(0), 1)).to(device2)
            if use_struct:
                GAB = GAB.to(device1)
                GAG = GAG.to(device1)
                zAB, zAG, gAB, gAG = encoder_model(GAB.x, GAB.edge_index, GAB.edge_attr, GAB.batch,\
                                                   GAG.x, GAG.edge_index, GAG.edge_attr, GAG.batch)
                gAB = encoder_model.encoderAB.project(gAB)
                gAG = encoder_model.encoderAG.project(gAG)
                h1 = torch.diag(torch.matmul(gAB, gAG.permute(1, 0))).unsqueeze(-1)
            if use_seq:
                tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
                tokenAB, tokenAG = tokenAB.to(device2), tokenAG.to(device2) 
                sAB, sAG = seq_encoder(tokenAB), seq_encoder(tokenAG)
                h2 = torch.diag(torch.matmul(sAB, sAG.permute(1, 0))).unsqueeze(-1)
                
                h2.to(device1)

            pred = torch.mean(torch.cat((h1,h2), dim=1), dim=1, keepdim=True)
            loss, __ = BinaryClass_Loss(pred, y)
            
            # for evaluation
            Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
            Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))

            epoch_loss += loss.item()
            
    result_df = metric(Y_pred.view(-1), Y_true.long().view(-1))
    return epoch_loss/len(dataloaderAB), result_df


def main(
    # data params
    train_data_dir="data/SabDab/train",
    test_data_dir="data/SabDab/test",
    device1="cuda:0",
    device2="cuda:0",
    # trainer params
    batch_size=32,
    test_batch_size=10,
    learning_rate=1e-3,
    epochs=50,
    early_stopping="valid_loss",
    # optimizer params
    optimizer="adamw",
    weight_decay=7.459343285726558e-05,
    learning_rate_scheduler="cosine",
    # model params
    use_struct=True,
    use_seq=False,
    pretrained_lm_model="Rostlab/prot_bert",
    hidden_dim=128,
    num_layers=12,
    temperature=0.2,
    # experiment params
    log_wandb=True,
    ):
    
    # find data length 
    if log_wandb:
        X = read_json(f'{train_data_dir}/XAb.json')
        config = {
          "num samples": len(X),
          "train batch size": batch_size,
          "test batch size": test_batch_size,
          "Learning Rate": learning_rate,
          "Learning Rate Scheduler": learning_rate_scheduler,  
          "Weight Decay": weight_decay,  
          "Early Stopping": early_stopping,  
          "epochs": epochs,
          "hidden dim": hidden_dim,
          "num layers": num_layers,
          "temperature": temperature,  
        }
        del X
        
        wandb.init(config=config, project="Bind CLassifier--GINECONV", entity="harsh1729")
        
    # preparing the data 
    train_data_lstAB, train_data_lstAG = prepare_dataABAG(f'{train_data_dir}/XAb.json', f'{train_data_dir}/XAg.json')
    train_loaderAB = DataLoader(train_data_lstAB[:200], batch_size=batch_size, shuffle=True)
    del train_data_lstAB
    
    node_dim = max(train_data_lstAG[0][0].x.size(-1), 1)
    edge_dim = max(train_data_lstAG[0][0].edge_attr.size(-1), 1)
    
    train_loaderAG = DataLoader(train_data_lstAG[:200], batch_size=batch_size, shuffle=True)
    del train_data_lstAG
  
    test_data_lstAB, test_data_lstAG = prepare_dataABAG(f'{test_data_dir}/XAb.json', f'{test_data_dir}/XAg.json')
    test_loaderAB = DataLoader(test_data_lstAB[-200:], batch_size=test_batch_size, shuffle=True)
    del test_data_lstAB
    test_loaderAG = DataLoader(test_data_lstAG[-200:], batch_size=test_batch_size, shuffle=True)
    del test_data_lstAG
    
    # Structure Model
    gconvAB = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device1)
    gconvAG = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device1)
    encoder_model = DualEncoder(encoderAB=gconvAB, encoderAG=gconvAG).to(device1)
    
    # Sequence Model
    seq_encoder = SeqEncoder().to(device2)
    tokenizer = BertTokenizer.from_pretrained(pretrained_lm_model, do_lower_case=False)
    
    if optimizer=="adamw":
        optimizer1 = optim.AdamW(encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer2 = optim.AdamW(seq_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer=="adam":
        optimizer1 = optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer2 = optim.Adam(seq_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # training
    with tqdm(total=epochs-1, desc='(T)') as pbar:
        for epoch in range(epochs):
            train_loss, train_result_df = train(encoder_model, seq_encoder, use_struct, use_seq,\
                                                tokenizer, train_loaderAB, train_loaderAG,\
                                                optimizer1, optimizer2, device1, device2)
            if log_wandb:
                wandb.log({"train BCE loss": train_loss,
                           "train precision": train_result_df['Precision'][0],
                           "train recall": train_result_df['Recall'][0], 
                           "train F1 score": train_result_df['F1 Score'][0]})

            pbar.set_postfix({'loss': train_loss})
            pbar.update()
    
    # testing
    test_loss, test_result_df = test(encoder_model, seq_encoder, use_struct, use_seq,\
                                     tokenizer, test_loaderAB, test_loaderAG, device1, device2)
    print(f"Test Loss = {test_loss}")
    if log_wandb:
        wandb.log({"test BCE loss": test_loss,
                   "test precision": test_result_df['Precision'][0],
                   "test recall": test_result_df['Recall'][0], 
                   "test F1 score": test_result_df['F1 Score'][0]})
    
    return test_result_df['F1 Score'][0]
