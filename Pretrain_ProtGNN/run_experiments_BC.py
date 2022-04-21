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


def train(model, tokenizer, dataloaderAB, dataloaderAG, optimizer, device_ids):
    model.train()
    
    epoch_loss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
        (GAB, seqAB, y) = dataAB
        (GAG, seqAG, y) = dataAG
        tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
        y = y.reshape(-1, 1).to(torch.float32)
        
        optimizer.zero_grad()
        
        pred = data_parallel(module=model, input=(GAB.cuda(), GAG.cuda(), tokenAB.cuda(), tokenAG.cuda()),\
                             device_ids=device_ids)
        # pred = model(GAB, GAG, tokenAB, tokenAG)
        pred = pred.to("cpu")
        
        loss, __ = BinaryClass_Loss(pred, y)
        
        loss.backward()
        optimizer.step()

        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
        
        epoch_loss += loss.item()
    
    result_df = metric(Y_pred.view(-1), Y_true.long().view(-1))
    return epoch_loss/len(dataloaderAB), result_df


def val(model, tokenizer, dataloaderAB, dataloaderAG, device_ids):
    model.eval()
    
    epoch_loss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
        (GAB, seqAB, y) = dataAB
        (GAG, seqAG, y) = dataAG
        tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
        y = y.reshape(-1, 1).to(torch.float32)

        pred = data_parallel(module=model, input=(GAB.cuda(), GAG.cuda(), tokenAB.cuda(), tokenAG.cuda()),\
                             device_ids=device_ids)  
        pred = pred.to("cpu")
        # pred = model(GAB, GAG, tokenAB, tokenAG)
        
        loss, __ = BinaryClass_Loss(pred, y)

        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
        
        epoch_loss += loss.item()
            
    result_df = metric(Y_pred.view(-1), Y_true.long().view(-1))
    return epoch_loss/len(dataloaderAB), result_df

def test(model, tokenizer, dataloaderAB, dataloaderAG, device_ids):
    model.eval()
    
    epoch_loss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for (dataAB, dataAG) in zip(cycle(dataloaderAB), dataloaderAG):
        (GAB, seqAB, y) = dataAB
        (GAG, seqAG, y) = dataAG
        tokenAB, tokenAG = tokenize(seqAB, tokenizer), tokenize(seqAG, tokenizer)
        y = y.reshape(-1, 1).to(torch.float32)

        pred = data_parallel(module=model, input=(GAB.cuda(), GAG.cuda(), tokenAB.cuda(), tokenAG.cuda()),\
                             device_ids=device_ids)  
        pred = pred.to("cpu")
        # pred = model(GAB, GAG, tokenAB, tokenAG)
        
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
    val_data_dir="data/SabDab/val"
    device_ids="[0]",
    # trainer params
    batch_size=32,
    test_batch_size=10,
    learning_rate=1e-7,
    epochs=50,
    early_stopping="valid_loss",
    # optimizer params
    weight_decay=7.459343285726558e-05,
    learning_rate_scheduler="cosine",
    # model params
    use_struct=True,
    use_seq=False,
    pretrained_lm_model="Rostlab/prot_bert",
    hidden_dim=128,
    num_layers=9,
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
    
    device_ids = [int(id) for id in device_ids.strip('][').split(', ')]
    
    # preparing the data 
    train_data_lstAB, train_data_lstAG = prepare_dataABAG(f'{train_data_dir}/XAb.json', f'{train_data_dir}/XAg.json')
    train_loaderAB = DataLoader(train_data_lstAB, batch_size=batch_size, shuffle=True)
    del train_data_lstAB
    
    node_dim = max(train_data_lstAG[0][0].x.size(-1), 1)
    edge_dim = max(train_data_lstAG[0][0].edge_attr.size(-1), 1)
    
    train_loaderAG = DataLoader(train_data_lstAG, batch_size=batch_size, shuffle=True)
    del train_data_lstAG

    val_data_lstAB, val_data_lstAG = prepare_dataABAG(f'{val_data_dir}/XAb.json', f'{val_data_dir}/XAg.json')
    val_loaderAB = DataLoader(val_data_lstAB, batch_size=test_batch_size, shuffle=True)
    del val_data_lstAB
    val_loaderAG = DataLoader(val_data_lstAG, batch_size=val_batch_size, shuffle=True)
    del val_data_lstAG
    
    test_data_lstAB, test_data_lstAG = prepare_dataABAG(f'{test_data_dir}/XAb.json', f'{test_data_dir}/XAg.json')
    test_loaderAB = DataLoader(test_data_lstAB, batch_size=test_batch_size, shuffle=True)
    del test_data_lstAB
    test_loaderAG = DataLoader(test_data_lstAG, batch_size=test_batch_size, shuffle=True)
    del test_data_lstAG
    
    # structure model
    gconvAB = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    gconvAG = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    encoder_model = DualEncoder(encoderAB=gconvAB, encoderAG=gconvAG)
    
    # sequence model
    seq_encoder = SeqEncoder()
    tokenizer = BertTokenizer.from_pretrained(pretrained_lm_model, do_lower_case=False)
    
    # combined model
    structseq_enc = StructSeqEnc(struct_enc=encoder_model, seq_enc=seq_encoder,\
                                use_struct=use_struct, use_seq=use_seq).cuda() 
    
    optimizer = optim.AdamW(structseq_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    with tqdm(total=epochs-1, desc='(T)') as pbar:
        for epoch in range(epochs):
            # training
            train_loss, train_result_df = train(structseq_enc, tokenizer, train_loaderAB,\
                                                train_loaderAG, optimizer, device_ids)
            if log_wandb:
                wandb.log({"train BCE loss": train_loss,
                           "train precision": train_result_df['Precision'][0],
                           "train recall": train_result_df['Recall'][0], 
                           "train F1 score": train_result_df['F1 Score'][0]})

            pbar.set_postfix({'loss': train_loss})
            pbar.update()
    
            # validation
            val_loss, val_result_df = val(structseq_enc, tokenizer, val_loaderAB, val_loaderAG, device_ids)
            print(f"val Loss = {val_loss}")
            if log_wandb:
                wandb.log({"val BCE loss": val_loss,
                           "val precision": val_result_df['Precision'][0],
                           "val recall": val_result_df['Recall'][0], 
                           "val F1 score": val_result_df['F1 Score'][0]})
            scheduler.step(val_loss)
                
        # testing
        val_loss, val_result_df = val(structseq_enc, tokenizer, val_loaderAB, val_loaderAG, device_ids)
        print(f"val Loss = {val_loss}")
        if log_wandb:
            wandb.log({"val BCE loss": val_loss,
                       "val precision": val_result_df['Precision'][0],
                       "val recall": val_result_df['Recall'][0], 
                       "val F1 score": val_result_df['F1 Score'][0]})
    
    return test_result_df['F1 Score'][0]
