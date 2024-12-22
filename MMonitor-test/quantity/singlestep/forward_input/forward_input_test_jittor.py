import jittor as jt
from jittor import nn, optim
import numpy as np
from MMonitor.quantity.singlestep import *
# 设置随机种子
jt.set_global_seed(42)
np.random.seed(42)

def if_similar(a,b,model,name,tolerance = 0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}当前{name}指标计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}指标预期值{b}")
    if abs(a - b) <= tolerance:
        print('True')
    else:
        print('False')
def test_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputMean(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'forward_input_mean'
    if_similar(quantity.get_output()[0],x_linear.mean(),model,name) 
def test_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputMean(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_mean'
    model = 'jittor_conv'
    if_similar(quantity.get_output()[0],x_conv.mean(),model,name)
def test_default_mean():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputMean(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_mean'
    model = 'jittor_bn'
    if_similar(quantity.get_output()[0],x_default.mean(),model,name)
def test_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputSndNorm(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_norm'
    model = 'linear'
    if_similar(quantity.get_output()[0],jt.norm(jt.flatten(x_linear)),model,name)
    
def test_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputSndNorm(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_norm'
    model = 'jittor_conv'
    if_similar(quantity.get_output()[0],jt.norm(jt.flatten(x_conv)),model,name)
def test_default_norm():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputSndNorm(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_norm'
    model = 'jittor_bn'
    if_similar(quantity.get_output()[0],jt.norm(jt.flatten(x_default)),model,name)
def test_linear_std():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputStd(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_std'
    model = 'jittor_linear'
    if_similar(quantity.get_output()[0],jt.std(x_linear),model,name)
def test_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputStd(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_std'
    model = 'jittor_conv'
    if_similar(quantity.get_output()[0],jt.std(x_conv),model,name)
def test_default_std():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputStd(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_std'
    model = 'jittor_bn'
    if_similar(quantity.get_output()[0],jt.std(x_default),model,name)
if __name__ == '__main__':
    test_linear_mean()
    test_conv_mean()
    test_default_mean()
    test_linear_norm()
    test_conv_norm()
    test_default_norm()
    test_linear_std()
    test_conv_std()
    test_default_std()