import torch
import numpy as np
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from regex import F
import torch
from torch import nn as nn
from MMonitor.quantity.singlestep import *

def if_similar(a,b,model,name):
    print(f"{model}的当前{name}指标计算值为{a}")
    print(f"{model}的{name}预期值为{b}")
    if np.allclose(a, b, rtol=1e-5, atol=1e-8):
        print("True")
    else:
        print("False")
def compute_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = ForwardInputMean(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    name = 'forward_input_mean'
    model = 'pytorch_linear'
    if_similar(float(x_linear.mean().item()),quantity_l.get_output()[0],model,name)
def compute_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = ForwardInputMean(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    name = 'forward_input_mean'
    model = 'pytorch_conv'
    if_similar(float(x_conv.mean().item()),quantity_c.get_output()[0],name,model)
def compute_default_mean():
    relu = nn.ReLU()
    x_default = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardInputMean(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x_default)
    i = 0
    quantity.track(i)
    # 反向传播
    y.sum().backward()
    name = 'forward_input_mean'
    model = 'pytorch_relu'
    if_similar(float(x_default.mean().item()),quantity.get_output()[0],model,name)
def compute_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = ForwardInputSndNorm(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    model = 'pytorch_linear'
    name='forward_input_norm'
    if_similar(float(x_linear.norm().item()),quantity_l.get_output()[0],model,name)
def compute_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = ForwardInputSndNorm(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    model = 'pytorch_conv'
    name='forward_input_norm'
    if_similar(float(x_conv.norm().item()),quantity_c.get_output()[0],model,name)
def compute_default_norm():
    relu = nn.ReLU()
    x_default = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardInputNorm(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x_default)
    i = 0
    quantity.track(i)
    # 反向传播
    y.sum().backward()
    model = 'pytorch_relu'
    name='forward_input_norm'
    if_similar(float(x_default.norm().item()),quantity.get_output()[0],model,name)
def compute_linear_std():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = ForwardInputStd(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    model = 'pytorch_linear'
    name='forward_input_std'
    if_similar(float(x_linear.std().item()),quantity_l.get_output()[0],model,name)
def compute_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = ForwardInputStd(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    model = 'pytorch_conv'
    name='forward_input_std'
    if_similar(float(x_conv.std().item()),quantity_c.get_output()[0],model,name)
def compute_default_std():
    relu = nn.ReLU()
    x_default = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardInputStd(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x_default)
    i = 0
    quantity.track(i)
    # 反向传播
    y.sum().backward()
    model = 'pytorch_relu'
    name='forward_input_std'
    if_similar(float(x_default.std().item()),quantity.get_output()[0],model,name)
compute_conv_mean()
compute_linear_mean()
compute_default_mean()
compute_conv_norm()
compute_linear_norm()
compute_default_norm()
compute_conv_std()
compute_linear_std()
compute_default_std()
    



