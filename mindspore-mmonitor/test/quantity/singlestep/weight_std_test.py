import mindspore as ms
from MMonitor.quantity.singlestep import *
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal

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
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    
    weight_std_direct = ops.std(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct))
def test_conv():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    
    weight_std_direct = ops.std(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct))
    
def test_default():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    optimizer(grads)
    quantity.track(i)
    
    weight_std_direct = ops.std(bn.gamma).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct))

test_linear()
test_conv()
test_default()