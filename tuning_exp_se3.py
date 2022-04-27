import argparse 
import os
import numpy as np
from train_se3 import main
import optuna
import torch

def objective(trial):
    config = dict.fromkeys(["learning_rate", "num_layers", "num_degrees",\
                            "num_channels", "num_nlayers", "div", "pooling", "head"])
    
    # device_ids(str -> list(int))
    device_ids = [int(k) for k in FLAGS.device_ids.strip('][').split(', ')]
    print(f"Available device ids : {device_ids}")
    
    # Generate the optimizers.
    lr_range = [float(k) for k in FLAGS.lr_range.strip('][').split(', ')]
    config["learning_rate"] = trial.suggest_float("lr", lr_range[0], lr_range[1])
    
    nlayers_range = [int(k) for k in FLAGS.nlayers_range.strip('][').split(', ')]
    config["num_layers"] = trial.suggest_int("num_layers", nlayers_range[0], nlayers_range[1])
    
    ndegrees_range = [int(k) for k in FLAGS.ndegrees_range.strip('][').split(', ')]
    config["num_degrees"] = trial.suggest_int("num_degrees", ndegrees_range[0], ndegrees_range[1])
    
    nchannels_range = [int(k) for k in FLAGS.nchannels_range.strip('][').split(', ')]
    config["num_channels"] = trial.suggest_int("num_channels", nchannels_range[0], nchannels_range[1])
    
    nnlayers_range = [int(k) for k in FLAGS.nnlayers_range.strip('][').split(', ')]
    config["num_nlayers"] = trial.suggest_int("num_nlayers", nnlayers_range[0], nnlayers_range[1])

    div_range = [int(k) for k in FLAGS.div_range.strip('][').split(', ')]
    config["div"] = trial.suggest_int("div", div_range[0], div_range[1])

    pool_lst = [str(k) for k in FLAGS.pool_lst.strip('][').split(', ')]
    config["pooling"] = trial.suggest_categorical("pooling", pool_lst)

    head_range = [int(k) for k in FLAGS.head_range.strip('][').split(', ')]
    config["head"] = trial.suggest_int("head", head_range[0], head_range[1])
    
    print(config)
    
    f1_score = main(# tuning params
                    FLAGS,
                    learning_rate=config["learning_rate"],
                    num_layers=config["num_layers"],
                    num_degrees=config["num_degrees"],
                    num_channels=config["num_channels"],
                    num_nlayers=config["num_nlayers"],
                    div=config["div"],
                    pooling=config["pooling"],
                    head=config["head"],
                    device_ids=device_ids
                    )
    return f1_score

if __name__ == "__main__":
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
    
    parser.add_argument('--nlayers_range', type=str, default="[4, 4]",
            help="Number of equivariant layers") #4
    parser.add_argument('--ndegrees_range', type=str, default="[4, 4]",
            help="Number of irreps {0,1,...,num_degrees-1}") #4
    parser.add_argument('--nchannels_range', type=str, default="[8, 14]",
            help="Number of channels in middle layers") #16
    parser.add_argument('--nnlayers_range', type=str, default="[1, 2]",
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div_range', type=str, default="[4, 4]",
            help="Low dimensional embedding fraction")
    parser.add_argument('--pool_lst', type=str, default="[avg, max]",
            help="Choose from avg or max")
    parser.add_argument('--head_range', type=str, default="[1, 1]",
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=1, 
            help="Batch size")
    parser.add_argument('--lr_range', type=str, default="[1e-5, 1e-3]", 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=100, 
            help="Number of epochs")
    parser.add_argument('--ntrials', type=int, default=20, 
            help="Number of optuna trials for tuning")

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
    parser.add_argument('--entity', type=str, default='maximentropy', 
            help="wandb account name")

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

    # Optimize!!!
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=FLAGS.ntrials)
