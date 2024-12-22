import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
import numpy as np
def if_similar(a,b,tolerance = 0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'当前计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False
def test_linear():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = WeightMean(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)
    optimizer.step(loss)
    quantity.track(i)
    # 添加直接计算权重均值的代码
    weight_mean_direct = jt.mean(l.weight).item()
    print(if_similar(quantity.get_output()[0],weight_mean_direct))
def test_conv():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = WeightMean(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_mean_direct = jt.mean(cov.weight).item()
    print(if_similar(quantity.get_output()[0],weight_mean_direct))
    
def test_default():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = WeightMean(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_mean_direct = jt.mean(bn.weight).item()
    print(if_similar(quantity.get_output()[0],weight_mean_direct))

test_linear()
test_conv()
test_default()