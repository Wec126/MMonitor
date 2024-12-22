
# 使用反向验证
import torch
import torch.nn as nn
from MMonitor.quantity.singlestep import * 
import numpy as np
# 设置随机种子
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def if_similar(a,b,model,name):
    print(f"{model}的{name}指标的当前计算值为{a}")
    print(f"{model}的{name}指标预期值为{b}")
    if np.allclose(a, b, rtol=1e-5, atol=1e-8):
        print("True")
    else:
        print("False")
def compute_linear_mean():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity = ForwardOutputMean(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    loss = y.sum()
    loss.backward()
    quantity.track(i)
    name = 'forward_output_mean'
    model = 'pytorch_linear'
    if_similar(quantity.get_output()[0],y.mean().item(),model,name)
def compute_default_mean():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardOutputMean(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    y.sum().backward()
    quantity.track(i)
    name = 'forward_output_mean'
    model = 'pytorch_relu'
    if_similar(quantity.get_output()[0],y.mean().item(),model,name)

def compute_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = ForwardOutputMean(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    y_c.sum().backward()
    quantity.track(i)
    name = 'forward_output_mean'
    model = 'pytorch_conv'
    if_similar(quantity.get_output()[0],y_c.mean().item(),model,name)
def compute_linear_norm():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity = ForwardOutputGradNorm(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_output_mean'
    if_similar(quantity.get_output()[0],y.norm().item(),model,name)
def compute_default_norm():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardOutputGradNorm(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    quantity.track(i)
    model = 'pytorch_relu'
    name = 'forward_output_mean'
    if_similar(quantity.get_output()[0],y.norm().item(),model,name)

def compute_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = ForwardOutputGradNorm(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_output_std'
    if_similar(quantity.get_output()[0],y_c.norm(2).item(),model,name)
def compute_linear_std():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity = ForwardOutputStd(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    quantity.track(i)
    name = 'forward_output_std'
    model = 'pytorch_linear'
    if_similar(quantity.get_output()[0],y.std().item(),model,name)
def compute_default_std():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = ForwardOutputStd(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    quantity.track(i)
    model = 'pytorch_relu'
    name = 'forward_output_mean'
    if_similar(quantity.get_output()[0],y.std().item(),model,name)

def compute_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = ForwardOutputStd(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_output_std'
    if_similar(quantity.get_output()[0],y_c.std().item(),model,name)

compute_linear_mean()
compute_conv_mean()
compute_default_mean()
compute_linear_norm()
compute_conv_norm()
compute_default_norm()
compute_linear_std()
compute_conv_std()
compute_default_std()



