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
from utils.utility_function import *
from scipy.linalg import svd  # Use SciPy's SVD function
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import shapiro
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
from copy import deepcopy
from methods.gpm import *
from methods.multi_task import *

# Importing modules related to your specific implementations
from models import *
import wandb

def train(args):
    global best_prec1
    rank = 0
    num_clients = 1
    
    print("Number of GPU available: ", torch.cuda.device_count())
    device = torch.device("cuda:{}".format(args.device))
    val_loader, bsz_val = test_Dataset_split(args, args.num_tasks)
    
    train_loader, bsz_train, cpt = partition_trainDataset(args, rank, num_clients, device, args.num_tasks)
    
    if args.baseline == 'gpm':
        client = gpm_client(rank, num_clients, device, args, train_loader, val_loader)
    elif args.baseline == 'multi_task':
        client = multi_task_client(rank, num_clients, device, args, train_loader, val_loader)
        
    client.bsz_train = bsz_train
    client.bsz_val = bsz_val
    
    ###############################################
    # For each task
    for task_id in range(args.num_tasks):
        if args.baseline == 'gpm': threshold = np.array([args.threshold] * client.no_layers) + task_id * np.array([args.increment_th] * client.no_layers) # threshold for GPM is increased for each task
        
        # For all node, initialize optimizer, scheduler
        # calculate feature matrix = M @ M^T
        client.init_task(task_id) 
                                 
        # For each local epoch, train the model on the local data
        for epoch in range(args.local_epochs):
            w, avg_loss = client.train_epoch()  # train on single epoch 
            client.scheduler.step()             # take gradient step 
            
            print_epoch = args.local_epochs // args.print_times
            if print_epoch == 0 : print_epoch = 5
            if ((epoch + 1) % print_epoch) == 0:
                precision, current_loss, current_acc = client.validate_seen_tasks()
                print("Epoch: {}, Task: {}, Avg train Loss: {:.4f}, Current Val Acc: {:.2f}, Accumulated Val Acc: {:.2f}".format(epoch, task_id, avg_loss, current_acc, precision))
                
                if args.wandb == True:
                    wandb.log({"Task": task_id, "Epoch": epoch, "Validation Acc": precision})
            else:
                print("Epoch: {}, Task: {}, Avg train Loss: {:.4f}".format(epoch, task_id, avg_loss))
            torch.cuda.empty_cache()
                    
        ###############################################
        # After all round and before proceeding to next task, perform GPM update of all clients (In our setting, update a GPM of single node)
        if args.baseline == 'gpm': client.myGPM_update(threshold) 
        elif args.baseline == 'multi_task': print("end of task ", task_id)
        
        client.print_performance
        accuracy_mat = client.update_acc_matrix()
        client.print_acc_matrix()
                    
    ###############################################
    # End of all task

    # Print final results
    acc = []
    bwt = []
    acc_val, bwt_val = client.print_final_results()
    acc.append(acc_val)
    bwt.append(bwt_val)
    
    print("average accuracy: ", np.mean(acc) , "average bwt: ", np.mean(bwt))   
    print()
    print(vars(args))
    # wandb finish