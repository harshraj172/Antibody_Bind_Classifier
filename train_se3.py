from utils.utils_profiling import * # load before other local modules

import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import math
import numpy as np
import torch
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import Antibody_Antigen_Dataset
import models as models
from models import *
from glob_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, criterion, dataloader, optimizer, scheduler, device_ids, FLAGS):
    model.train()
  
    num_iters = len(dataloader)
    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        tokenAB, tokenAG = tokenize(seqAB), tokenize(seqAG)
        tokenAB = tokenAB.to(device)
        tokenAG = tokenAG.to(device)
        gAB = gAB.to(device)
        gAG = gAG.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)

        loss = criterion(pred, y)
        rloss += loss
        
        # backprop
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch + i / num_iters)
        
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] train Loss/step: {loss:.5f}")
        if FLAGS.use_wandb:
            wandb.log({"train loss/step": to_np(loss)})

        if FLAGS.profile and i == 10:
            sys.exit()

        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
    rloss /= FLAGS.train_size
    results_df = metric(Y_true.reshape(-1), Y_pred.reshape(-1))
    
    print(f"...[{epoch}|train] Loss: {rloss:.5f} [units]")
    print(results_df)
    if FLAGS.use_wandb:
        wandb.log({"train BCE loss": to_np(rloss),
                   "train precision": results_df['Precision'][0],
                   "train recall": results_df['Recall'][0], 
                   "train F1 Score": results_df['F1 Score'][0]})

def val_epoch(epoch, model, criterion, dataloader, device_ids, FLAGS):
    model.eval()

    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        tokenAB, tokenAG = tokenize(seqAB), tokenize(seqAG)
        tokenAB = tokenAB.to(device)
        tokenAG = tokenAG.to(device)
        gAB = gAB.to(device)
        gAG = gAG.to(device)
        y = y.to(device)

        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)
        
        loss = criterion(pred, y)
        rloss += loss
        
        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
    rloss /= FLAGS.val_size
    results_df = metric(Y_true.reshape(-1), Y_pred.reshape(-1))
    
    print(f"...[{epoch}|val] Loss: {rloss:.5f} [units]")
    print(results_df)
    if FLAGS.use_wandb:
        wandb.log({"val BCE loss": to_np(rloss),
                   "val precision": results_df['Precision'][0],
                   "val recall": results_df['Recall'][0], 
                   "val F1 score": results_df['F1 Score'][0]})
    return results_df['F1 Score'][0]

def test_epoch(epoch, model, criterion, dataloader, device_ids, FLAGS):
    model.eval()

    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        tokenAB, tokenAG = tokenize(seqAB), tokenize(seqAG)
        tokenAB = tokenAB.to(device)
        tokenAG = tokenAG.to(device)
        gAB = gAB.to(device)
        gAG = gAG.to(device)
        y = y.to(device)

        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)
        
        loss = criterion(pred, y)
        rloss += loss
        
        # for evaluation
        Y_true = torch.cat((Y_true.to('cpu'), y.to('cpu')))
        Y_pred = torch.cat((Y_pred.to('cpu'), pred.to('cpu')))
    rloss /= FLAGS.test_size
    results_df = metric(Y_true.reshape(-1), Y_pred.reshape(-1))
    
    print(f"...[{epoch}|test] Loss: {rloss:.5f} [units]")
    print(results_df)
    if FLAGS.use_wandb:
        wandb.log({"test BCE loss": to_np(rloss),
                   "test precision": results_df['Precision'][0],
                   "test recall": results_df['Recall'][0], 
                   "test F1 Score": results_df['F1 Score'][0]})


def collate(samples):
    structseqAB_lst, structseqAG_lst, y = map(list, zip(*samples))
    batched_graphAB = dgl.batch([s['struct'] for s in structseqAB_lst])
    batched_graphAG = dgl.batch([s['struct'] for s in structseqAG_lst])
    seqAB_lst = [s['seq'] for s in structseqAB_lst]
    seqAG_lst = [s['seq'] for s in structseqAG_lst]
    return batched_graphAB, seqAB_lst, batched_graphAG, seqAG_lst, torch.tensor(y)


def main(FLAGS,
         learning_rate,
         num_layers,
         num_degrees,
         num_channels,
         num_nlayers,
         div,
         pooling,
         head,
         device_ids,):

    # wandb logging
    config = {
      "num_layers": num_layers,
      "num_degrees": num_degrees,
      "num_channels": num_channels,
      "num_nonliner_layers": num_nlayers,
      "head": head,
      "batch size": FLAGS.batch_size,
      "learning_rate": learning_rate,
      "epochs": FLAGS.num_epochs,
    }
    
    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{num_degrees}-l{num_layers}-{num_channels}-{num_nlayers}'
    
    if FLAGS.use_wandb:
        # Log all args to wandb
        if FLAGS.name:
            wandb.init(config=config, project=f'{FLAGS.wandb}', name=f'{FLAGS.name}', entity=f'{FLAGS.entity}')
        else:
            wandb.init(config=config, project=f'{FLAGS.wandb}', entity=f'{FLAGS.entity}')    
    
    # Prepare data
    train_dataset = Antibody_Antigen_Dataset(f'{FLAGS.train_data_dir}/XAb.json', f'{FLAGS.train_data_dir}/XAg.json')
    train_loader = DataLoader(train_dataset, 
                              batch_size=FLAGS.batch_size, 
                              shuffle=True, 
                              collate_fn=collate, 
                              num_workers=FLAGS.num_workers,
                              drop_last=True)

    val_dataset = Antibody_Antigen_Dataset(f'{FLAGS.val_data_dir}/XAb.json', f'{FLAGS.val_data_dir}/XAg.json') 
    val_loader = DataLoader(val_dataset, 
                            batch_size=FLAGS.batch_size, 
                            shuffle=False, 
                            collate_fn=collate, 
                            num_workers=FLAGS.num_workers,
                            drop_last=True)

    test_dataset = Antibody_Antigen_Dataset(f'{FLAGS.test_data_dir}/XAb.json', f'{FLAGS.test_data_dir}/XAg.json') 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, 
                             shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers,
                             drop_last=True)

    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)
    FLAGS.test_size = len(test_dataset)

    # Choose model
    model = models.__dict__.get(FLAGS.model)(FLAGS.use_struct, 
                                             FLAGS.use_seq,
                                             num_layers, 
                                             train_dataset.node_feature_size, 
                                             train_dataset.edge_feature_size,
                                             num_channels=num_channels,
                                             num_nlayers=num_nlayers,
                                             num_degrees=num_degrees,
                                             div=div,
                                             pooling=pooling,
                                             n_heads=head,
                                             pretrained_lm_model=FLAGS.pretrained_lm_model, 
                                             pretrained_lm_dim=FLAGS.pretrained_lm_dim)


    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    #wandb.watch(model)

    # Optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # Loss
    criterion = torch.nn.BCELoss(reduction='sum')

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')
    
    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, criterion, train_loader, optimizer, scheduler, device_ids, FLAGS)
        f1_score = val_epoch(epoch, model, criterion, val_loader, device_ids, FLAGS)
        test_epoch(epoch, model, criterion, test_loader, device_ids, FLAGS)
    
    return f1_score
