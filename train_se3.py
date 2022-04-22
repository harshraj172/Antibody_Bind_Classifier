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

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, criterion, dataloader, optimizer, scheduler, device_ids, FLAGS):
    model.train()
  
    num_iters = len(dataloader)
    rloss = 0
    Y_true, Y_pred = torch.tensor([]), torch.tensor([])
    for i, (gAB, seqAB, gAG, seqAG, y) in enumerate(dataloader):
        tokenAB, tokenAG = tokenize(seqAB), tokenize(seqAG)
        tokenAB = tokenAB.cuda()
        tokenAG = tokenAG.cuda()
        gAB = gAB.to("cuda")
        gAG = gAG.to("cuda")
        y = y.cuda()

        optimizer.zero_grad()
        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)
        # pred = data_parallel(module=model, input=(gAB, tokenAB, gAG, tokenAG),\
        #                      device_ids=device_ids)

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
        tokenAB = tokenAB.cuda()
        tokenAG = tokenAG.cuda()
        gAB = gAB.to("cuda")
        gAG = gAG.to("cuda")
        y = y.cuda()

        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)
        # pred = data_parallel(module=model, input=(gAB, tokenAB, gAG, tokenAG),\
        #                      device_ids=device_ids)
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
        tokenAB = tokenAB.cuda()
        tokenAG = tokenAG.cuda()
        gAB = gAB.to("cuda")
        gAG = gAG.to("cuda")
        y = y.cuda()

        # run model forward and compute loss
        pred = model(gAB, tokenAB, gAG, tokenAG)
        # pred = data_parallel(module=model, input=(gAB, tokenAB, gAG, tokenAG),\
        #                      device_ids=device_ids)
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
            wandb.init(config=config, project=f'{FLAGS.wandb}', name=f'{FLAGS.name}', entity="harsh1729")
        else:
            wandb.init(config=config, project=f'{FLAGS.wandb}', entity="harsh1729")    
    
    # Prepare data
    train_dataset = Antibody_Antigen_Dataset(f'{FLAGS.train_data_dir}/XAb.json', f'{FLAGS.train_data_dir}/XAg.json')
    train_loader = DataLoader(train_dataset, 
                              batch_size=FLAGS.batch_size, 
                              shuffle=True, 
                              collate_fn=collate, 
                              num_workers=FLAGS.num_workers)

    val_dataset = Antibody_Antigen_Dataset(f'{FLAGS.val_data_dir}/XAb.json', f'{FLAGS.val_data_dir}/XAg.json') 
    val_loader = DataLoader(val_dataset, 
                            batch_size=FLAGS.batch_size, 
                            shuffle=False, 
                            collate_fn=collate, 
                            num_workers=FLAGS.num_workers)

    test_dataset = Antibody_Antigen_Dataset(f'{FLAGS.test_data_dir}/XAb.json', f'{FLAGS.test_data_dir}/XAg.json') 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, 
                             shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers)

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
    model = model.cuda()
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


def run_exp():
    parser = argparse.ArgumentParser()\
    
    parser.add_argument('--use_seq', type=eval, default=False,
            help="Use sequence info of protein")
    parser.add_argument('--use_struct', type=eval, default=True,
            help="Use structure info of protein")
    
    # Model parameters
    parser.add_argument('--model', type=str, default='StructSeqEnc', 
            help="String name of model")
    parser.add_argument('--pretrained_lm_model', type=str, default="Rostlab/prot_bert",
            help="Pretrained LM model name")
    parser.add_argument('--pretrained_lm_dim', type=int, default=1024,
            help="Pretrained LM model out dim")
    
    parser.add_argument('--nlayers_range', type=str, default="[2, 6]",
            help="Number of equivariant layers") #4
    parser.add_argument('--ndegrees_range', type=str, default="[2, 6]",
            help="Number of irreps {0,1,...,num_degrees-1}") #4
    parser.add_argument('--nchannels_range', type=str, default="[8, 16]",
            help="Number of channels in middle layers") #16
    parser.add_argument('--nnlayers_range', type=str, default="[1, 4]",
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div_range', type=str, default="[2, 4]",
            help="Low dimensional embedding fraction")
    parser.add_argument('--pool_lst', type=str, default="[avg, max]",
            help="Choose from avg or max")
    parser.add_argument('--head_range', type=str, default="[1, 4]",
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=1, 
            help="Batch size")
    parser.add_argument('--lr_range', type=str, default="[1e-5, 1e-3]", 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=150, 
            help="Number of epochs")

    # Data
    parser.add_argument('--train_data_dir', type=str, default='data/SabDab/train',
            help="training data directory Antibodies")
    parser.add_argument('--val_data_dir', type=str, default='data/SabDab/test',
            help="validation data directory Antibodies")
    parser.add_argument('--test_data_dir', type=str, default='data/SabDab/test',
            help="validation data directory Antibodies")


    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--use_wandb', type=eval, default=True,
            help="To use wandb or not - [True, False]")
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='Bind-Classifier', 
            help="wandb project name")

    # Miscellaneas
    parser.add_argument('--device_ids', type=str, default="[0]", 
            help="The cuda device id")
    parser.add_argument('--num_workers', type=int, default=0, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # device_ids(str -> list(int))
    device_ids = [int(k) for k in FLAGS.device_ids.strip('][').split(', ')]
    
    f1_score = main(
                 FLAGS,
                 learning_rate=1e-3,
                 num_layers=2,
                 num_degrees=4,
                 num_channels=4,#16
                 num_nlayers=1,
                 div=4,
                 pooling="avg",
                 head=1,
                 device_ids=device_ids,
                 )
    return f1_score

run_exp()
