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
from dataset import _Antibody_Antigen_Dataset, Antibody_Antigen_Dataset

import models #as models

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()

    num_iters = len(dataloader)
    for i, (gAB, gAG, y) in enumerate(dataloader):
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(gAB, gAG)
        l1_loss, __, rescale_loss = loss_fnc(pred, y)

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]")
        if FLAGS.use_wandb:
            if i % FLAGS.log_interval == 0:
                wandb.log({"Train L1 loss": to_np(l1_loss), 
                        "Rescale loss": to_np(rescale_loss)})

        if FLAGS.profile and i == 10:
            sys.exit()
    
        scheduler.step(epoch + i / num_iters)

def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, (gAB, gAG, y) in enumerate(dataloader):
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(gAB, gAG).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.val_size

    print(f"...[{epoch}|val] rescale loss: {rloss:.5f} [units]")
    if FLAGS.use_wandb:
        wandb.log({"Val L1 loss": to_np(rloss)})

def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, (gAB, gAG, y) in enumerate(dataloader):
        gAB = gAB.to(FLAGS.device)
        gAG = gAG.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(gAB, gAG).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.test_size

    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")
    if FLAGS.use_wandb:
        wandb.log({"Test L1 loss": to_np(rloss)})


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def collate(samples):
    graphsAB, graphsAG, y = map(list, zip(*samples))
    batched_graphAB = dgl.batch(graphsAB)
    batched_graphAG = dgl.batch(graphsAG)
    return batched_graphAB, batched_graphAG, torch.tensor(y)

def main(FLAGS, UNPARSED_ARGV):

    # Prepare data
    train_dataset = Antibody_Antigen_Dataset(FLAGS.meta_data_address, 
                                                     mode='train', 
                                                     transform=RandomRotation())
    train_loader = DataLoader(train_dataset, 
                              batch_size=FLAGS.batch_size, 
			                        shuffle=True, 
                              collate_fn=collate, 
                              num_workers=FLAGS.num_workers)

    val_dataset = Antibody_Antigen_Dataset(FLAGS.meta_data_address, 
                                                   mode='valid') 
    val_loader = DataLoader(val_dataset, 
                            batch_size=FLAGS.batch_size, 
			                      shuffle=False, 
                            collate_fn=collate, 
                            num_workers=FLAGS.num_workers)

    test_dataset = Antibody_Antigen_Dataset(FLAGS.meta_data_address, 
                                                    mode='test') 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, 
			                       shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)
    FLAGS.test_size = len(test_dataset)

    # Choose model
    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, 
                                             train_dataset.atom_feature_size, 
                                             FLAGS.num_channels,
                                             num_nlayers=FLAGS.num_nlayers,
                                             num_degrees=FLAGS.num_degrees,
                                             edge_dim=train_dataset.num_bonds,
                                             div=FLAGS.div,
                                             pooling=FLAGS.pooling,
                                             n_heads=FLAGS.head)
    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)
    #wandb.watch(model)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Loss function
    def task_loss(pred, target, use_mean=True):
	eps = 1e-10
        l1_loss = criterion(pred+eps, target)
        l2_loss = torch.sum(torch.abs(pred - target))
        if use_mean:
            l2_loss /= pred.shape[0]

        rescale_loss = l1_loss # train_dataset.norm2units(l1_loss, FLAGS.task)
        return l1_loss, l2_loss, rescale_loss
      
    # Loss function
#     def task_loss(pred, target, use_mean=True):
#         l1_loss = torch.sum(torch.abs(pred - target))
#         l2_loss = torch.sum((pred - target)**2)
#         if use_mean:
#             l1_loss /= pred.shape[0]
#             l2_loss /= pred.shape[0]

#         rescale_loss = train_dataset.norm2units(l1_loss, FLAGS.task)
#         return l1_loss, l2_loss, rescale_loss

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        val_epoch(epoch, model, task_loss, val_loader, FLAGS)
        test_epoch(epoch, model, task_loss, test_loader, FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='Dual_SE3Transformer', 
            help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
            help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
            help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16,
            help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0,
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=8, 
            help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, 
            help="Number of epochs")

    # Data
    parser.add_argument('--meta_data_address', type=str, default='/content/meta_file.csv',
            help="Address to structure file")
    parser.add_argument('--task', type=str, default='BindClassification',
            help="task ['BindClassification']")

    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--use_wandb', type=eval, default=False,
            help="To use wandb or not - [True, False]")
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='equivariant-attention', 
            help="wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if FLAGS.use_wandb:
      # Log all args to wandb
      if FLAGS.name:
          wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}')
      else:
          wandb.init(project=f'{FLAGS.wandb}')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
