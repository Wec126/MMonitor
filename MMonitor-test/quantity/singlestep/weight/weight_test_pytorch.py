import numpy as np
from MMonitor.quantity.singlestep import *
import torch
from torch import nn as nn
# 设置随机种子
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def if_similar(a,b,model,name):
    print(f'{model}的{name}指标的计算所得值:{a}')
    print(f"{model}的{name}的预期指标：{b}")
    if np.allclose(a, b, rtol=1e-5, atol=1e-8):
        print("True")
    else:
        print("False")
def compute_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightMean(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_mean_direct = l.weight.mean().item()
    model = 'pytorch_linear'
    name = 'weight_mean'
    if_similar(quantity_l.get_output()[0].item(), weight_mean_direct,model,name)
def compute_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightMean(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_mean_direct = cov.weight.mean().item()
    model = 'pytorch_conv'
    name = 'weight_mean'
    if_similar(quantity_c.get_output()[0].item(),weight_mean_direct,model,name)
def compute_default_mean():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightMean(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_mean_direct = bn.weight.mean().item()
    model = 'pytorch_bn'
    name = 'weight_mean'
    if_similar(quantity.get_output()[0].item(),weight_mean_direct,model,name)
def compute_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightNorm(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_norm_direct = l.weight.norm().item()
    model = 'pytorch_linear'
    name = 'weight_norm'
    if_similar(quantity_l.get_output()[0].item(), weight_norm_direct,model,name)
def compute_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightNorm(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_norm_direct = cov.weight.norm().item()
    model = 'pytorch_conv'
    name = 'weight_norm'
    if_similar(quantity_c.get_output()[0].item(),weight_norm_direct,model,name)
def compute_default_norm():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightNorm(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_norm_direct = bn.weight.norm().item()
    model = 'pytorch_bn'
    name = 'weight_norm'
    if_similar(quantity.get_output()[0].item(),weight_norm_direct,model,name)
def compute_linear_std():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightStd(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_std_direct = l.weight.std().item()
    model = 'pytorch_linear'
    name = 'weight_std'
    if_similar(quantity_l.get_output()[0].item(), weight_std_direct,model,name)
def compute_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightStd(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_std_direct = cov.weight.std().item()
    model = 'pytorch_conv'
    name = 'weight_std'
    if_similar(quantity_c.get_output()[0].item(),weight_std_direct,model,name)
def compute_default_std():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightStd(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_std_direct = bn.weight.std().item()
    model = 'pytorch_bn'
    name = 'weight_std'
    if_similar(quantity.get_output()[0].item(),weight_std_direct,model,name)
compute_linear_mean()
compute_conv_mean()
compute_default_mean()
compute_linear_norm()
compute_conv_norm()
compute_default_norm()
compute_linear_std()
compute_conv_std()
compute_default_std()
