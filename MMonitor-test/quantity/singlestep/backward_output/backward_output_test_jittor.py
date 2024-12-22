import numpy as np
import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
jt.set_global_seed(42)
np.random.seed(42)
def if_similar(a,b,model,name,tolerance = 0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}的{name}指标的当前计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}指标预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False
def test_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputMean(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))
def test_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputMean(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))

def test_default_mean():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputMean(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))
def test_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)   
    model = 'jittor_linear'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))
def test_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))

def test_default_norm():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))
def test_linear_std():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputStd(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)   
    name = 'backward_output_std'
    model = 'jittor_linear' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))
def test_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputStd(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    name = 'backward_output_std'
    model = 'jittor_conv' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))

def test_default_std():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputStd(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    name = 'backward_output_std'
    model = 'jittor_linear' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))
test_linear_mean()
test_conv_mean()
test_default_mean()
test_linear_norm()
test_conv_norm()
test_default_norm()
test_linear_std()
test_conv_std()
test_default_std()