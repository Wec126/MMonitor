import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
import numpy as np
def is_similar(a, b, tolerance=0.1):
    # 检查基本性质
    print(f"当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) < tolerance
# 在文件开头添加
def setup_seed(seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
def test_linear():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0], 1))

def test_conv():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0],1))

def test_default():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0], 1))

test_linear()
test_conv()
test_default()
