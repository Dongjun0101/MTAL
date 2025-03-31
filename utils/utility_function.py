
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

# Importing modules related to your specific implementations
from models import *
import wandb

def format_val(x):
    """
    If x is really an integer (type int or a float with x.is_integer()==True),
    print an integer. Otherwise, print with 4 decimal places.
    """
    # If the object is actually an int
    if isinstance(x, int):
        return str(x)
    # If it's a float but effectively an integer, e.g. 3.0
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    # Otherwise, treat it like a float with 4 decimal digits
    return f"{float(x):.4f}"

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i] = indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size = int(skew * class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices

class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""

    def __init__(self, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        assert classes % tasks == 0
        self.data = data
        self.partitions = {}
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=8)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices_full = sort_index.tolist()
        task_data_len = int(data_len / tasks)

        for n in range(tasks):
            ind_per_task = indices_full[n * task_data_len: (n + 1) * task_data_len]
            indices_rand, indices = skew_sort(ind_per_task, skew=skew, classes=int(classes / tasks),
                                              class_size=class_size, seed=seed)
            self.partitions[n] = []
            for frac in sizes:
                if skew == 1:  # completely non iid
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices[0:part_len])
                    indices = indices[part_len:]
                elif skew == 0:  # iid setting
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices_rand[0:part_len])
                    indices_rand = indices_rand[part_len:]  # remove to use full data at each node for experiment
                else:  # partially non-iid
                    part_len = int(frac * task_data_len * skew);
                    part_len_rand = int(frac * task_data_len * (1 - skew))
                    part_ind = indices[0:part_len] + indices_rand[0:part_len_rand]
                    self.partitions[n].append(part_ind)
                    indices = indices[part_len:]
                    indices_rand = indices_rand[part_len_rand:]

    def use(self, partition, task):
        return Partition(self.data, self.partitions[task][partition])

class DataPartition_5set(object):
    """ Partitions 5-datasets across different nodes, not setup for non-IID data yet, works only for SKEW=0"""
    def __init__(self, data_type, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        #assert classes%tasks==0
        self.data = data
        self.partitions = {}
        indices_full = []
        data_len= []

        for i in range(len(data)):
            dataset = torch.utils.data.DataLoader(data[i], batch_size=512, shuffle=False, num_workers=2)
            data_len.append(len(data[i]))
            labels= []

            if(data_type=='5datasets'):
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    labels = labels+targets.tolist()
            else:
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    t = np.array(targets.tolist()).reshape(-1)
                    labels = labels+t.tolist()

            sort_index = np.argsort(np.array(labels))
            indices_full.append(sort_index.tolist())

        for n in range(tasks):
            task_data_len = int(data_len[n])
            ind_per_task = indices_full[n]
            rng = Random()
            rng.seed(seed)
            rng.shuffle(ind_per_task)
            self.partitions[n] = []
            for frac in sizes:
                part_len = int(frac*task_data_len)
                self.partitions[n].append(ind_per_task[0:part_len])
                ind_per_task = ind_per_task[part_len:] #remove to use full data at each node for experiment

    def use(self, partition, task):
        return Partition(self.data[task], self.partitions[task][partition])

def partition_trainDataset(args, rank, size, device, tasks=2):
    """Partitioning dataset"""
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # mnist have one channel only
        classes = 10
        class_size = {x: 6000 for x in range(10)}  # each class has 6000 sample images
        dataset = datasets.MNIST(root=f'data_mnist', train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        classes = 10
        class_size = {x: 5000 for x in range(10)}  # creates a dictionary where each class (from 0 to 9) is assigned a size of 5000.
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=True, transform=transforms.Compose([  # dataset loading
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        classes = 100
        class_size = {x: 500 for x in range(100)}
        c = int(classes / tasks)  # 100 classes / tasks
        dataset = datasets.CIFAR100(root=f'data_cifar100', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    elif args.dataset == '5datasets':
        dataset= []
        classes= 10 #each task has 10 classes
        c= int(classes)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'Five_data/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,) 
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'Five_data/',train=True,download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(), transforms.Normalize(mean,std)]))
        
        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'Five_data/SVHN',split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'Five_data/', train=True, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))

        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'Five_data/notmnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    elif args.dataset == 'miniimagenet':
        dataset= []
        classes= 100 #each task has 5 classes
        c = int(classes/tasks)
        class_size = {x:500 for x in range(100)} 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset= miniimagenet_setup.MiniImageNet(root='../QGPM_baseline/miniimagenet', train=True, transform=transforms.Compose([transforms.Resize((84,84)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        
        
    train_set = {}

    bsz = int((args.batch_size) / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]  # For 4 nodes, partition_size = [0.25, 0.25, 0.25, 0.25]

    if (rank == 0):
        print("Data partition_sizes among clients:", partition_sizes)

    # partition object creation
    # normalized entire dataset, partition size among the nodes, skewness(0.0 for iid, 1.0 for non-iid), classes, class size, seed, device, tasks
    if(args.dataset=='5datasets'):
        partition= DataPartition_5set(args.dataset, dataset, partition_sizes, skew=args.skew, classes=classes, class_size=0, seed=args.seed, device=device, tasks=tasks)
    else:
        partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device, tasks=tasks)
    
    # partition for continual learning
    for n in range(tasks):
        task_partition = partition.use(rank, n)  # partition for corresponding rank(node) / n-th task
        train_set[n] = torch.utils.data.DataLoader(task_partition, batch_size=bsz, shuffle=True, num_workers=8, drop_last=(size == 8))  # train_set for n-th task

    return train_set, bsz, c # c = class per task

def test_Dataset_split(args, tasks):
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        dataset = datasets.MNIST(root=f'data_mnist', train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=f'data_cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == '5datasets':
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'Five_data/',train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,)
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'Five_data/',train=False, download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False)
        for image, target in loader:
            image=image.expand(1,3,image.size(2),image.size(3))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'Five_data/SVHN',split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'Five_data/', train=False, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))
        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'Five_data/notmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

        val_set={}
        val_bsz = 64

        for n in range(tasks):
            task_data = dataset[n]
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5) #shuffle=False gives low test acc for bn with track_run_stats=False
    elif args.dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dataset= miniimagenet_setup.MiniImageNet(root='../QGPM_baseline/miniimagenet', train=False, transform=transforms.Compose([transforms.Resize((84,84)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels+targets.tolist()

        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len/tasks)
        val_set={}
        val_bsz=10

        for n in range(tasks):
            ind_per_task = indices[n*task_data_len: (n+1)*task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=2)
     
     
    if (args.dataset != '5datasets') and (args.dataset != 'miniimagenet'):
        val_set = {}
        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=5)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len / tasks)
        val_bsz = 64

        for n in range(tasks):
            ind_per_task = indices[n * task_data_len: (n + 1) * task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5)

    return val_set, val_bsz

def average_weights(w): # w : list of weights, weight = node.model.state_dict()
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res