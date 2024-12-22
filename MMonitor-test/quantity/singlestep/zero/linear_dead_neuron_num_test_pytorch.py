from regex import F
import torch
from torch import nn as nn

from MMonitor.quantity.singlestep import *
import numpy as np

def if_similar(a,b,model,name):
    print(f"{model}的{name}指标的当前计算指标为{a}")
    print(f"{model}的{name}指标的预期指标为{b}")
    if a == b:
        print('True')
    else:
        print('False')
def compute_linear():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = LinearDeadNeuronNum(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    output = y.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():  # 使用torch.all()替代np.all()
            dead_count += 1
    quantity_l.track(i)
    model = 'pytorch_linear'
    name = 'linear_dead_neuron_num'
    if_similar(quantity_l.get_output()[0].item(),dead_count,model,name)

def compute_conv():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = LinearDeadNeuronNum(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_conv)
    output = y_c.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():
            dead_count += 1
    quantity_c.track(i)
    model = 'pytorch_conv'
    name = 'linear_dead_neuron_num'
    if_similar(quantity_c.get_output()[0].item(),dead_count,model,name)

def compute_default():
    relu = nn.ReLU()
    x_default = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = LinearDeadNeuronNum(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x_default)
    i = 0
    quantity.track(i)
    output = y.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():  # 将 np.all 改为 torch.all
            dead_count += 1
    # 反向传播
    y.sum().backward()
    model = 'pytorch_bn'
    name = 'linear_dead_neuron_num'
    if_similar(quantity.get_output()[0].item(),dead_count,model,name)


compute_linear()
compute_conv()
compute_default()
    



