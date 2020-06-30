#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# sys.path.insert(0, '')
sys.path.append('../graph/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from mlp import MLP
from activation import activation_factory
from tools import k_adjacency, normalize_adjacency_matrix


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2 #这个数值一直是0
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1), #（3，1），（5，1）
                                dilation=(self.window_dilation, 1), #（1，1）
                                stride=(self.window_stride, 1),     #（1，1）
                                padding=(self.padding, 0))          #（0，0）

    def forward(self, x):
        # Input shape: (N,C,T,V)
        N, C, T, V = x.shape
        #输入数据4维
        x = self.unfold(x)
        #unfold函数的输入数据是四维，但输出是三维的，沿T维度将window_size大小的T维度拉伸成一维
        #输出维度：(N,windowsize*V,C*(T-windowsize+1))
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)#将window_size大小的时间维度上的组成一个
        return x


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,  #A的阶次
                 window_size,
                 disentangled_agg=True,#邻接矩阵的两种模式，是否分解
                 use_Ares=True,#是否使用自适应的Ares矩阵
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)#在时间轴上拼成一个更大的邻接矩阵，                 #大小为（N*window_size,N*window_size）

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)] #with_self=True,
            #意味着加上I,但前面已经包括了I?
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales]) #正则化
            #在列方向上进行拼接，大小为（N*window_size*num_scales,N*window_size)
        else: #是先正则化还是先求幂
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)] #num_scales理论上应该的等于windowsize
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)
        #上式是用A的多项式来计算
        self.A_scales = torch.Tensor(A_scales)#转换成pytorch中的tensor
        self.V = len(A_binary)#len是第一个维度上的长度

        if use_Ares:#自适应矩阵的参数化
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        #残差连接？
        if not residual:
            self.residual = lambda x: 0#将x投影为0
        elif (in_channels == out_channels):
            self.residual = lambda x: x#保持不变
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)#激活函数
    #下面函数是在原来的邻接矩阵上扩大相应的时间步（window-size倍）
    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size#没有作用
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        #np.title:沿x轴和y轴复制
        #函数作用是创建一个更大的邻接矩阵
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows 

        # Build graphs
        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)#将变量放到GPU上
        #最终的A,这里的A不是方阵，大小为（N*window_size*num_scales,N*window_size)
        # Perform Graph Convolution
        res = self.residual(x)
        agg = torch.einsum('vu,nctu->nctv', A, x)#爱因斯坦简记法，按维度u求和（节点）
        agg = agg.view(N, C, T, self.num_scales, V)#这里是将前面concatenate的矩阵分开?
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)#特征数增加num_scales倍
        out = self.mlp(agg)
        out += res
        return self.act(out)

