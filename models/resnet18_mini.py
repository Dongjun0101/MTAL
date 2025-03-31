
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
import numpy as np
from copy import deepcopy
from collections import OrderedDict

__all__ = ['ResNet18_mini', 'get_Rmatrix_resnet18_mini']

def get_bucket_and_sign(col_idx, n_buckets, seed_offset=0):
    """
    Given a column index and the number of buckets, return the bucket index and sign
    for that column in a reproducible way (using a seeded RNG).
    """
    rng = np.random.RandomState(seed=col_idx + seed_offset)
    bucket = rng.randint(0, n_buckets)
    sign = rng.choice([-1, 1])
    return bucket, sign

def streaming_count_sketch_column(col_j, col_idx, n_buckets, Y, seed_offset=0):
    # Get the bucket index and sign for this column
    b, s = get_bucket_and_sign(col_idx, n_buckets, seed_offset)

    # Accumulate into the correct bucket
    # Y[:, b] += s * col_j
    Y[:, b] = Y[:, b] + s * col_j

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()

    def forward(self, x):
        self.act['conv_0'] = x
        out = relu(self.bn1(self.conv1(x))) 
        self.act['conv_1'] = out
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet_mini(nn.Module):
    def __init__(self, block, num_blocks, nf, dataset, ntasks=2):
        super(ResNet_mini, self).__init__()
        self.in_planes = nf
        if(dataset=='5datasets' or dataset=='medmnist'):
            self.conv1  = conv3x3(3, nf * 1, 1)
            factor = 4
        else:       # miniimagenet
            self.conv1  = conv3x3(3, nf * 1, 2)
            factor = 9
        self.bn1    = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.ntasks = ntasks
        self.linear=torch.nn.ModuleList()
        for t, n in self.ntasks:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * factor, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.act['conv_in'] = x
        out = relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.ntasks:
            y.append(self.linear[t](out))
        return y

def ResNet18_mini(dataset, ntasks, nf=32):
    return ResNet_mini(BasicBlock, [2, 2, 2, 2], nf, dataset, ntasks)


def get_Rmatrix_resnet18_mini(net, device, data, nodes, rank, dataset): 
    # Collect activations by forward pass
    net.eval()
    if(data.size(dim=1)==1):
        data= data.repeat(1, 3, 1, 1)
    out  = net(data)
    
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    del net.act['conv_in']
    del net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1']
    del net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1']
    del net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1']
    del net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']

    batch_list  = [10,10,10,10, 10,10,10,10, 50,50,50,100,100, 100,100,100,100] #scaled

    # network arch 
    if(dataset=='5datasets' or dataset=='medmnist'):
        stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
        map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4]
    else:
        stride_list = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
        map_list    = [84, 42,42,42,42, 42,21,21,21, 21,11,11,11, 11,6,6,6] 

    if(dataset=='medmnist'):
        in_channel  = [ 3, 32,32,32,32, 32,64,64,64, 64,128,128,128, 128,256,256,256] 
    else:
        in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 

    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final   = [] # list containing GPM Matrices 
    mat_list    = []
    mat_sc_list = []
    for i in range(len(stride_list)):
        ksz = 3 
        bsz = batch_list[i]
        st  = stride_list[i]     
        k   = 0
        s   = compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        Y = np.zeros((ksz*ksz*in_channel[i],ksz*ksz*in_channel[i]))
        col_index = 0
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()

        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    # mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    column = act[kk, : , st*ii : ksz + st*ii, st*jj : ksz + st*jj].reshape(-1)
                    
                    # # (i) Gaussian random projection
                    # np.random.seed(col_index)
                    # random_vector = np.random.randn(ksz*ksz*in_channel[i])
                    # Y = Y + np.outer(column, random_vector)
                    # col_index += 1
                    
                    # (ii) Sparse random projection (count sketch)
                    streaming_count_sketch_column(column, col_index, ksz*ksz*in_channel[i], Y)
                    col_index += 1
                    
        mat_list.append(Y)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            Y = np.zeros((1*1*in_channel[i],1*1*in_channel[i]))
            col_index = 0
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        # mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        column = act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        
                        # (i) Gaussian random projection
                        # np.random.seed(col_index)
                        # random_vector = np.random.randn(1*1*in_channel[i])
                        # Y = Y + np.outer(column, random_vector)
                        # col_index += 1
                        
                        # (ii) Sparse random projection (count sketch)
                        streaming_count_sketch_column(column, col_index, 1*1*in_channel[i], Y)
                        col_index += 1
                        
            mat_sc_list.append(Y) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    if(rank==0):
        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_final)):
            print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print('-'*30)
    return mat_final    




def print_model_size(model):
    param_size, param_count = 0, 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()  # in bytes

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size
    
    # 천 단위 콤마를 적용해 출력
    print(f"Model Parameters      : {param_count:,}")
    print(f"Parameter Size       : {param_size:,} bytes")
    print(f"Buffer Size          : {buffer_size:,} bytes")
    print(f"Total Model Size     : {total_size:,} bytes "
          f"({total_size / 1024**2:,.2f} MB)")


# def print_gpm_sizes(mat_final):
#     """
#     Given the list of representation matrices from get_Rmatrix_resnet18,
#     compute and print the GPM dimension and size for each layer, then
#     print the total GPM size.
#     """
#     total_gpm_size = 0
#     for idx, mat in enumerate(mat_final, start=1):
#         # 'mat' has shape (#features, #samples)
#         feature_dim = mat.shape[0]
#         gpm_size = feature_dim * feature_dim
#         total_gpm_size += gpm_size
#         print(f"Layer {idx}: Representation shape = {mat.shape}, "
#               f"GPM dimension = {feature_dim}, GPM size = {gpm_size}")
#     print("-" * 50)
#     print(f"Total GPM size across all layers = {total_gpm_size}\n")

    
if __name__ == "__main__":
    # Example 'taskcla' with 10 tasks, each having 1 class for simplicity
    taskcla = [(t, 1) for t in range(10)]
    
    # Build the model
    net = ResNet18_mini('5dtasets', taskcla, nf=20)
    
    # Print parameter count and memory usage
    print_model_size(net)