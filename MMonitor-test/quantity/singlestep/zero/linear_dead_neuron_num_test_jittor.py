import jittor as jt
from jittor import nn, optim
import numpy as np
from MMonitor.quantity.singlestep import *
def if_similar(a,b,model,name):
    print(f"{model}的{name}指标当前计算指标为{a}")
    print(f"{model}的{name}指标预期指标为{b}")
    if a == b:
        return True
    else:
        return False
def test_linear():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = LinearDeadNeuronNum(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    # 手动计算死亡神经元
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx]) < 1e-6):  # 设置阈值
            dead_count += 1
    model = 'jittor_linear'
    name = 'linear_dead_neuron_num'
    print(if_similar(quantity.get_output()[0],dead_count,model,name))

    
def test_conv():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = LinearDeadNeuronNum(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx]) < 1e-6):  # 设置阈值
            dead_count += 1
    model = 'jittor_conv'
    name = 'linear_dead_neuron_num'
    print(if_similar(quantity.get_output()[0],dead_count,model,name))

def test_default():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = LinearDeadNeuronNum(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx]) < 1e-6):  # 设置阈值
            dead_count += 1
    model = 'jittor_bn'
    name = 'linear_dead_neuron_num'
    print(if_similar(quantity.get_output()[0],dead_count,model,name))

test_linear()
test_conv()
test_default()