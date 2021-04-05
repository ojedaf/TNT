import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model.temporalShiftModule.ops.models import TSN
# import task_generator as tg
# import os
# import math
# import argparse
# import random

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, conf):
        super(CNNEncoder, self).__init__()
        self.TSM = TSN(conf['num_class'], conf['num_segments'], conf['modality'],
                base_model=conf['arch'],
                consensus_type=conf['consensus_type'],
                dropout=conf['dropout'],
                img_feature_dim=conf['img_feature_dim'],
                partial_bn=not conf['no_partialbn'],
                pretrain=conf['pretrain'],
                is_shift=conf['shift'], shift_div=conf['shift_div'], shift_place=conf['shift_place'],
                fc_lr5=conf['fc_lr5'],
                temporal_pool=conf['temporal_pool'],
                non_local=conf['non_local'], get_emb = True)
        self.reduce_channels = conf['reduce_channs']
        self.pooling = conf['pooling']
        print("reduce_chan: ", self.reduce_channels)
        if self.reduce_channels:
            self.conv2d = nn.Conv2d(in_channels=512, out_channels=conf['chann_dim'], kernel_size = 1)
        
    def forward(self,x):
        out = self.TSM(x)

        # Reduce channel dim
        if self.reduce_channels:
            out_size = out.size()
            out = out.view(-1, out_size[2], out_size[3], out_size[4])
            out = self.conv2d(out)
            out = out.view(out_size[0], out_size[1], out.size(1), out_size[3], out_size[4])
        
        # BxTxCxHxW
        if self.pooling != 'attention':
            # BxTxCxHxW -----> BxCxTxHxW
            out = out.permute(0, 2, 1, 3, 4)
            # out (B, C, T, H, W)
        return out # 64


class Covpool3d(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         t = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = t*h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         t = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = t*h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,t,h,w)
         return grad_input

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(2,64,kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttModule(nn.Module):
    def __init__(self, ratio, temp_dim, emb):
        super(AttModule, self).__init__()
        out_temp_dim = int(temp_dim/ratio)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(temp_dim, out_temp_dim),
            nn.ReLU(),
            nn.Linear(out_temp_dim, temp_dim)
        )

        self.maxPool = nn.MaxPool1d(emb, stride=emb)
        self.avgPool = nn.AvgPool1d(emb, stride=emb)
    
    def forward(self, x):

        x_max = self.maxPool(x)
        x_max = self.mlp(x_max)

        x_avg = self.avgPool(x)
        x_avg = self.mlp(x_avg)

        x_sum = x_max + x_avg

        scale = F.sigmoid(x_sum)

        return scale.unsqueeze(2)

def CovpoolLayer(var):
    return Covpool3d.apply(var)


def PowerNorm(var, slope=1):
    return (1-torch.exp(slope*var))/(1+torch.exp(slope*var))