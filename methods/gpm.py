# copied from codec samely
from models import *
import os
import numpy as np
import copy
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.utility_function import *

def update_GPM (self, representation_matrix, threshold):
    epsilon = 1e-10

    if self.rank == 0 : print ('Threshold of GPM: ', threshold) 
    
    number_of_added_components = []
    if self.QGPM == []: # First task
        for i in range(len(representation_matrix)): # i = 0, 1, 2, 3, 4
            activation = torch.Tensor(representation_matrix[i]).to(self.device)
            try: 
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            except Exception as e:
                try:
                    U, S, Vh = torch.linalg.svd(activation + epsilon, full_matrices=False)
                except Exception as e2:
                    raise e2
            
            # criteria
            sval_total = (S.pow(2)).sum()
            sval_ratio = (S.pow(2))/sval_total
            cumulative = torch.cumsum(sval_ratio.cpu(), dim=0).to(sval_ratio.device)
            r = int((cumulative < threshold[i]).sum().item())
                        
            self.QGPM.append(U[:,0:r])
                    
            number_of_added_components.append(r)

    else:           # subsequent task
        for i in range(len(representation_matrix)):
            activation = torch.Tensor(representation_matrix[i]).to(self.device)
            try: 
                U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False) # ERROR! it says Activation contain nan or inf value
            except Exception as e:
                try:
                    U1, S1, Vh1 = torch.linalg.svd(activation + epsilon, full_matrices=False)
                except Exception as e2:
                    raise e2
                    
            sval_total = (S1.pow(2)).sum()
            
            # Projected Representation 
            act_hat_torch = activation - torch.mm(self.feature_mat[i], activation) # orthogonal components with respect to existing GPM            act_hat = act_hat.astype(np.float64)
            try : 
                U,S,Vh = torch.linalg.svd(act_hat_torch, full_matrices=False)
            except Exception as e:
                try:
                    U, S, Vh = torch.linalg.svd(act_hat_torch + epsilon, full_matrices=False)
                except Exception as e2:
                    raise e2

            # criteria
            sval_hat = (S.pow(2)).sum()
            sval_ratio = (S.pow(2))/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
                
            if r == 0:
                if self.rank == 0 : print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                number_of_added_components.append(r)
                continue

                    
            newGPM = torch.cat((self.QGPM[i],U[:,0:r]), dim=1)                          # Concatenate the new feature with the existing GPM
            if newGPM.shape[1] > newGPM.shape[0] : self.QGPM[i]=newGPM[:,0:newGPM.shape[0]]
            else                                 : self.QGPM[i]=newGPM   
                             
            number_of_added_components.append(r)


    if self.rank == 0:
        N = []
        K = []
        print ('Number of Added Components: ', number_of_added_components)
        print('-' * 40)
        
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.QGPM)):
            print ('Layer {} : {}/{}'.format(i+1, self.QGPM[i].shape[1], self.QGPM[i].shape[0]))
            N.append(self.QGPM[i].shape[0])
            K.append(self.QGPM[i].shape[1])
        print('-' * 40)
        
        # full precision gpm
        mem = 0
        for n, k in zip(N, K):
            mem = mem + n*k*32 # bits
        print("full precision gpm memory overhead: ", mem/8/1024/1024, "MB")
        print('-' * 40)

# The Node class encapsulating the node's functionality
class gpm_client:
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
        self.QGPM = []                                 # current node's feature list ; i.e. GPM
        self.feature_mat = []                         # M @ M^T
        self.importance_scaled_feature_mat = []                  # M @ D @ importance-D @ M^T
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
        self.calculate_feature_mat()

    def calculate_feature_mat(self):
        self.feature_mat = []               # used at gpm construction
        self.importance_scaled_feature_mat = []  

        GPM = [torch.Tensor(copy.deepcopy(self.QGPM[i])).to(self.device) for i in range(len(self.QGPM))]
        # Update feature_mat based on GPM
        self.feature_mat = [] # M @ M^T
        self.importance_scaled_feature_mat = []   

        if (self.task_id > 0):
            for i in range(len(GPM)):
                ith_feature_mat = GPM[i] @ GPM[i].T
                self.feature_mat.append(ith_feature_mat)
                self.importance_scaled_feature_mat.append(ith_feature_mat)
        
        if self.rank == 0 : print('-' * 40)
                           
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
            
            # Apply GPM constraints if needed
            if (self.task_id > 0) :
                kk = 0
                if self.args.arch == 'alexnet':
                    for k, (m, params) in enumerate(self.model.named_parameters()):
                        if k < 15 and len(params.size()) != 1:
                            sz = params.grad.data.size(0)
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), self.importance_scaled_feature_mat[kk]).view(params.size())
                            kk += 1
                        elif (k < 15 and len(params.size()) == 1) and self.task_id != 0:
                            params.grad.data.fill_(0)
                elif self.args.arch == 'resnet':
                    for k, (m,params) in enumerate(self.model.named_parameters()):
                        if len(params.size())==4:
                            sz =  params.grad.data.size(0)
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                self.importance_scaled_feature_mat[kk]).view(params.size())
                            kk+=1
                        elif len(params.size())==1 and self.task_id !=0:
                            params.grad.data.fill_(0)

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

    def myGPM_update(self, threshold):
        count, data_in = 0, None
        train_loader = self.train_loaders_list[self.task_id] # using own local data
        
        # Collect training sub-sample data for GPM update
        for i, (input, target) in enumerate(train_loader):
            inp, target_in = Variable(input).to(self.device), Variable(target).to(self.device)
            
            if inp.size(1) == 1:  # If grayscale, repeat to make 3 channels
                inp = inp.repeat(1, 3, 1, 1)
            
            data_in = torch.cat((data_in, inp), 0) if data_in is not None else inp
            count += target_in.size(0)
            if count >= 100: break
            
        # compute local representation matrix(activation)
        
        # representation_matrix = get_representation_matrix(self.model, self.device, data_in, 4, self.rank) # 4 : layer count. defined at alexnet_model.py
        if (self.args.arch == 'resnet') and (self.args.dataset == '5datasets' or self.args.dataset == 'cifar100'):
            # The last Boolean parameter is for 'nodes' in your snippet; set False if not distributed
            representation_matrix = get_Rmatrix_resnet18_cifar100(
                self.model,
                device=self.device,
                data=data_in,
                nodes=False,
                rank=self.rank,
                dataset=self.args.dataset
            )
        
        elif (self.args.arch == 'resnet') and (self.args.dataset == 'miniimagenet'):    
            representation_matrix = get_Rmatrix_resnet18_mini(
                self.model,
                device=self.device,
                data=data_in,
                nodes=False,
                rank=self.rank,
                dataset=self.args.dataset
            )
            
        elif self.args.arch == 'alexnet':
            representation_matrix = get_representation_matrix(
                self.model,
                self.device,
                data_in,
                4,           # the layer count for alexnet
                self.rank
            )
        
        for i in range(len(representation_matrix)):
            activation = representation_matrix[i]
            has_nan = np.isnan(activation).any()
            has_inf = np.isinf(activation).any()
            if has_nan or has_inf: 
                if has_nan: 
                    nan_count = np.isnan(activation).sum()
                    nan_indices = np.argwhere(np.isnan(activation))
                elif has_inf: 
                    inf_count = np.isinf(activation).sum()
                    inf_indices = np.argwhere(np.isinf(activation))
                raise ValueError('Node {} Task {} Layer {} - Activation matrix has nan or inf value, {} of nan or inf value starting at {}'.format(self.rank, self.task_id, i + 1, nan_count, nan_indices[0] ))
                sys.exit(1)
                
        # update local GPM using local data
        update_GPM(self, representation_matrix, threshold)

    def print_final_results(self):
        print('Node {} Final Avg Accuracy: {:5.2f}%'.format(self.rank, self.acc_matrix[-1].mean()))
        bwt = np.mean((self.acc_matrix[-1] - np.diag(self.acc_matrix))[:-1])
        print('Node {} Backward Transfer: {:5.2f}%'.format(self.rank, bwt))
        
        return self.acc_matrix[-1].mean() , bwt