import mindspore as ms
from MMonitor.quantity.singlestep import *
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
# 添加更完整的随机种子设置
import random
random.seed(42)
np.random.seed(42)
ms.set_seed(42)
def if_similar(a,b,model,name,tolerance = 0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}的{name}指标的当前计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}的预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False
def test_linear_mean():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))
def test_conv_mean():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))
    
def test_default_mean():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(bn.gamma).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))

def test_linear_norm():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'weight_norm'
    weight_norm_direct = ops.norm(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))
def test_conv_norm():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'weight_norm'
    weight_norm_direct = ops.norm(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))
    
def test_default_norm():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_norm'
    
    weight_norm_direct = ops.norm(bn.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))

def test_linear_std():
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
    model = 'mindspore_linear'
    name = 'weight_std'
    weight_std_direct = ops.std(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))
def test_conv_std():
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
    model = 'mindspore_conv'
    name = 'weight_std'
    weight_std_direct = ops.std(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))
    
def test_default_std():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_std'
    
    weight_std_direct = ops.std(bn.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))

test_linear_mean()
test_conv_mean()
test_default_mean()
test_linear_norm()
test_conv_norm()
test_default_norm()
test_linear_std()
test_conv_std()
test_default_std()