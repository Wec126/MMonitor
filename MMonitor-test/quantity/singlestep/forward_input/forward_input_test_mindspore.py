import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
from MMonitor.quantity.singlestep import *
import numpy as np
import random

# 添加更完整的随机种子设置
random.seed(42)
np.random.seed(42)
ms.set_seed(42)

# 确保使用确定性的计算模式
ms.context.set_context(mode=ms.context.PYNATIVE_MODE)

def forward_hook_fn(cell,grad_input,grad_output):
    inputs = grad_input[0]
    setattr(cell, 'input', inputs) 
def if_similar(a, b, model,name,tolerance=0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}当前的{name}计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}指标预期值{b}")
    if abs(a - b) <= tolerance:
        print('True')
    else:
        print('False')
def test_linear_mean():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputMean(l)
    
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name='forward_input_mean'
    if_similar(quantity.get_output()[0], ops.mean(x_linear),model,name)
    handle.remove()
def test_conv_mean():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputMean(conv)
    
    handle = conv.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name='forward_input_mean'
    if_similar(quantity.get_output()[0], ops.mean(x_conv),model,name)
    handle.remove()
def test_default_mean():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputMean(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    handle = bn.register_forward_hook(forward_hook_fn)    
    # loss_fn = nn.MSELoss()  # 均方误差
    # optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)

    # # 6. 进行前向和反向传播
    # model_with_loss = nn.WithLossCell(bn, loss_fn)
    # train_step = nn.TrainOneStepCell(model_with_loss, optimizer)
    # epoch = 1
    # loss = train_step(x_default, target)
    # quantity.track(epoch)
    def forward_fn():
        return bn(x_default)
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_default)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name='forward_input_mean'
    if_similar(quantity.get_output()[0], np.mean(x_default.asnumpy()),model,name)
    handle.remove()
def test_linear_std():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputStd(l)
    
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name='forward_input_norm'
    if_similar(quantity.get_output()[0], ops.std(x_linear),model,name)
    handle.remove()
def test_conv_std():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputStd(conv)
    
    handle = conv.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i) 
    model = 'mindspore_conv'
    name='forward_input_norm'
    if_similar(quantity.get_output()[0], ops.std(x_conv),model,name)
    handle.remove()
def test_default_std():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputStd(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    handle = bn.register_forward_hook(forward_hook_fn)    
    def forward_fn():
        return bn(x_default)
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_default)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name='forward_input_norm'
    if_similar(quantity.get_output()[0], np.std(x_default.asnumpy()),model,name)
    handle.remove()
def test_linear_norm():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputNorm(l)
    
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'forward_input_norm'
    if_similar(quantity.get_output()[0], ops.norm(x_linear),model,name)
    handle.remove()
def test_conv_norm():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputNorm(conv)
    
    handle = conv.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        y = forward_fn()
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_input_norm'
    if_similar(quantity.get_output()[0], ops.norm(x_conv),model,name)
    handle.remove()
def test_default_norm():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardInputNorm(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    handle = bn.register_forward_hook(forward_hook_fn)    
    loss_fn = nn.MSELoss()  # 均方误差
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)

    # 6. 进行前向和反向传播
    model_with_loss = nn.WithLossCell(bn, loss_fn)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for epoch in range(1):  # 单轮训练示例
        loss = train_step(x_default, target)
        quantity.track(epoch)
        model = 'mindspore_bn'
        name = 'forward_input_norm'
        if_similar(quantity.get_output()[0], np.linalg.norm(x_default.asnumpy()),model,name)
    handle.remove()
test_linear_mean()
test_conv_mean()
test_default_mean()
test_linear_norm()
test_conv_norm()
test_default_norm()
test_linear_std()
test_conv_std()
test_default_std()
