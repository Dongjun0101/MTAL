# copied from codec samely
from models import *
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
from utils.utility_function import *


# The Node class encapsulating the node's functionality
class multi_task_client:
    # Constructor
    def __init__(self, rank, size, device, args, train_loaders_list, val_loader): 
        self.rank = rank    # node's rank
        self.size = size    # total number of nodes; i.e. network size
        self.device = device    # device(GPU) ID
        self.args = args        # passed arguments
        self.train_loaders_list = train_loaders_list  # n-th node train_loader[k-th task]
        self.val_loader = val_loader                  # val_loader[k-th task] ; shared across the all nodes
        if args.dataset == '5datasets':
            self.task_details = [(task, 10) for task in range(args.num_tasks)] # (# of task, # of classes per task)
            self.cpt = 10  # Classes per task
        else:
            self.task_details = [(task, int(args.classes / args.num_tasks)) for task in range(args.num_tasks)] # [(0, 10), (1, 10), ..., (9, 10)] )
            self.cpt = int(args.classes / args.num_tasks) # classes per task
        self.model = self.init_model()                # current node's model
        self.optimizer = self.init_optimizer()        # current node's optimizer
        self.scheduler = self.init_scheduler()        # current node's scheduler
        if args.arch == "alexnet":
            self.no_layers = 5                            # Number of layers to consider for GPM
        elif args.arch == "resnet":
            self.no_layers = 20
        self.criterion = nn.CrossEntropyLoss()
        self.best_prec1 = 0
        self.bsz_train = None                         # train batch size; Will be set when data is partitioned
        self.bsz_val = None                           # validation batch size; Will be set when val_loader is created
        self.acc_matrix = np.zeros((args.num_tasks, args.num_tasks)) # current node's accuracy matrix
    
    def init_model(self): # used at constructor
        # torch.manual_seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        if self.args.arch == 'alexnet':
            model = alexnet(self.task_details)

        elif self.args.arch == 'alex_quarter':
            model = alexnet_scaled(self.task_details)
            
        elif (self.args.arch == 'resnet') and (self.args.dataset == '5datasets' or self.args.dataset == 'cifar100'):
            # You can pass your dataset name, the “task_details”, and choose nf=32 or 64, etc.
            # model = ResNet18(self.args.dataset, self.task_details, nf=32)
            model = ResNet18_cifar100(self.task_details, nf=20)
            
        elif (self.args.arch == 'resnet') and (self.args.dataset == 'miniimagenet'):
            model = ResNet18_mini(self.args.dataset, self.task_details, nf=20)

        else:
            raise ValueError("Unknown architecture")
        model.to(self.device)
        return model

    def init_optimizer(self): # used at init_task
        optimizer = optim.SGD(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=False)
        return optimizer

    def init_scheduler(self): # used at init_task
        gamma = 0.1
        step1 = int(self.args.local_epochs / 2)
        step2 = int(3 / 4 * self.args.local_epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=[step1, step2])
        return scheduler

    def init_task(self, task_id): # called for every task
        self.task_id = task_id
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
           
    def train_epoch(self): # For every epoch
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.train() # switch to train mode
        # step = len(self.train_loaders_list[self.task_id]) * self.bsz_train * epoch # num of mini batch * batch size(num of data per MB) * epoch = total num of data processed until now
        
        for batch_idx, (input, target) in enumerate(self.train_loaders_list[self.task_id]): # For every mini-batch
            input_var, target_var = input.to(self.device), (target % self.cpt).to(self.device) # moves the input batch input to the specified device (e.g., GPU), storing it in input_var
            if input_var.size(1) == 1: # if input is grayscale, expand it to 3 channels
                input_var = input_var.repeat(1,3,1,1)
            
            outputs = self.model(input_var)
            output = outputs[self.task_id]
            loss = self.criterion(output, target_var) 
            self.optimizer.zero_grad() # zero the gradient buffers
            loss.backward()            # calculate the gradient of the loss w.r.t. the model parameters

            self.optimizer.step() # take gradient step
            
            output = output.float()
            loss = loss.float()
            
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # step += self.bsz_train
            
        return self.model.state_dict(), losses.avg # return the model parameter after update / average loss

    def validate_seen_tasks(self):
        prec = []
        total_data_num = 0
        for tn in range(self.task_id + 1): # upto current task. assume we are on task 2, then tn = 0,1,2
            acc, loss = self.validate_task(tn)
            total_data_num += len(self.val_loader[tn])
            prec.append(acc * len(self.val_loader[tn]))
            if tn == self.task_id:
                current_loss = loss
                current_acc = acc
        acc_prev_tasks = sum(prec) / total_data_num
        return acc_prev_tasks, current_loss, current_acc

    def validate_task(self, task_id):
        val_loader = self.val_loader[task_id]
        top1 = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input_var, target_var = input.to(self.device), (target % self.cpt).to(self.device)
                
                if input_var.size(1) == 1:
                    input_var = input_var.repeat(1, 3, 1, 1)
                
                outputs = self.model(input_var)
                output = outputs[task_id]
                loss = self.criterion(output, target_var)
                output = output.float()
                loss = loss.float()
                prec1 = accuracy(output.data, target_var)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
        return top1.avg, losses.avg

    def update_acc_matrix(self):
        for tn in range(self.task_id + 1):
            acc, loss = self.validate_task(tn)
            self.acc_matrix[self.task_id, tn] = acc
            
        return self.acc_matrix

    def print_acc_matrix(self):
        print('Node {} Overall Accuracies:'.format(self.rank))
        for i_a in range(self.task_id + 1):
            print('\t', end='')
            for j_a in range(self.acc_matrix.shape[1]):
                print('{:5.1f}% '.format(self.acc_matrix[i_a, j_a]), end='')
            print()

    def print_performance(self):
        prec1 = self.validate_seen_tasks()
        print('Node {} Task {} Accuracy: {:5.2f}%'.format(self.rank, self.task_id, prec1))

    def print_final_results(self):
        print('Node {} Final Avg Accuracy: {:5.2f}%'.format(self.rank, self.acc_matrix[-1].mean()))
        bwt = np.mean((self.acc_matrix[-1] - np.diag(self.acc_matrix))[:-1])
        print('Node {} Backward Transfer: {:5.2f}%'.format(self.rank, bwt))
        
        return self.acc_matrix[-1].mean() , bwt