# QGPM baseline code
# experimental result : https://docs.google.com/spreadsheets/d/1c1IWlcCGB1lhRWJwGoV9nEV7RenY1wBILW40JgO_Uak/edit?usp=sharing


import argparse
import os
import time
import numpy as np
import copy
import sys
import random
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
from math import ceil
from random import Random
from utils import notmnist_setup
from utils import miniimagenet_setup
from scipy.linalg import svd  # Use SciPy's SVD function
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import shapiro
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
from copy import deepcopy
from trainer import train

# Importing modules related to your specific implementations
from models import *
import wandb

def str2bool(v):
    return v.lower() in ("true")

# Argument parsing  dd
parser = argparse.ArgumentParser(description='Proper AlexNet for CIFAR10/CIFAR100 in pytorch')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--biased', dest='biased', action='store_true', help='biased compression')
parser.add_argument('--unbiased', dest='biased', action='store_false', help='biased compression')
parser.add_argument('--level', default=32, type=int, metavar='k', help='quantization level 1-32')
parser.add_argument('--compress', default=False, type=str2bool, metavar='COMP', help='True: compress by sending coefficients associated with the orthogonal basis space')
parser.add_argument('--device', default=0, type=int, help='GPU device ID')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--port', dest='port', help='between 3000 to 65000', default='29500', type=str)  # 29500
parser.add_argument('--method', default='svd', type=str, help='gram or svd')
parser.add_argument('--skew', default=0.0, type=float, help='belongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--n_clients', default=1, type=int, help='total number of nodes; 10')  # number of nodes. 
parser.add_argument('--frac', default=1, type=float, help='fraction of client to be updated')  # 1.0
# You don't need to care about the above parameters



# You need to care about the below parameters
parser.add_argument('--momentum', default=0.0, type=float, metavar='M', help='momentum; resnet = 0.9 / alexnet = 0.0')
parser.add_argument('--increment_th', default=0.001, type=float, help='increase threshold linearly across tasks; 0.001 -> alexnet, 0 -> resnet') # default = 0.001
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate, alexnet -> 0.01') # default = 0.01
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--deterministic', default=True, type=str2bool)  # deterministic behavior

# For Alexnet -> CIFAR100 -> SET '--classes' = 100, '--num_tasks' = 10, '--batch-size' = 128, '--local_epochs' = 50, , '--lr' = 0.01
# For Resnet -> 5datasets -> SET '--classes' = 10, '--num_tasks' = 5, '--batch-size' = 128, '--local_epochs' = 50, , '--lr' = 0.01
# For Resnet -> miniimagenet -> SET '--classes' = 100, '--num_tasks' = 20, '--batch-size' = 32, '--local_epochs' = 50, , '--lr' = 0.01
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet', help='alexnet / resnet')
parser.add_argument('--dataset', dest='dataset', default='cifar100', help='available datasets: cifar100, 5datasets, miniimagenet', type=str)  # make sure to check the number of classes below
parser.add_argument('--classes', default=100, type=int, help='number of classes in the dataset')  # miniimagenet : 100, cifar100 : 100, mnist and cifar10 : 10
parser.add_argument('--num_tasks', default=10, type=int, help='number of tasks (over time)')  # CIFAR-100 split into 10 tasks, Each task having 10 classes
parser.add_argument('--print_times', default=5, type=int)

parser.add_argument('--wandb', default=False, type=str2bool)  # Wandb enable
parser.add_argument('--baseline', default='gpm', help='gpm, agem, derpp, er, fdr, multi_task', type=str)  # multi_task : no continual learning scheme; act as a lower bound
parser.add_argument('--seed', default=1, type=int, help='set seed, defualt = 1234')
parser.add_argument('--local_epochs', default=1, type=int)          # 2 local epoch / 150 global round or 5 local epoch / 60 global round 

parser.add_argument('--buffer_size', default=70, type=int, help='Replay buffer capacity')           # defualt = 6000

# gpm parameters
parser.add_argument('--threshold', default=0.6, type=float, help='threshold for the gradient memory; 0.9 -> alexnet, 0.965-> resnet')  # Similar to GPM-Codebase, default = 0.9



args = parser.parse_args()

def wandb_initialization():                         
    wandb.init(
        project="MTAL_Project",
        
        name=f"{args.baseline}, {args.arch}, {args.dataset}, Seed: {args.seed}",

        config=args
    )
    
def set_seed(seed):
    # Set the random seed for Python's random module
    random.seed(seed)
    # Set the random seed for NumPy
    np.random.seed(seed)
    # Set the random seed for PyTorch (CPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you are using CUDA, set the seed for all CUDA GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    if args.deterministic==True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    print(f"Random seed set to: {seed}")

def main():
    
    if ((args.dataset == 'cifar100') and (args.classes == 10)) or (((args.dataset == 'cifar10') or (args.dataset == 'mnist')) and (args.classes == 100)):
        sys.exit("Dataset and classes mismatch")
        
    if args.wandb:
        wandb_initialization()
        
    print(vars(args))
    
    set_seed(args.seed)
    
    train(args)
    
    if args.wandb:
        wandb.finish()
        
###########################################################
# Main execution
if __name__ == '__main__':
    main()