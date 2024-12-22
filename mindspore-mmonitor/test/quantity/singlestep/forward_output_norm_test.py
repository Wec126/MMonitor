from turtle import forward
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
from MMonitor.quantity.singlestep import *
import numpy as np
def forward_hook_fn(cell,grad_input,grad_output):
    output = grad_output
    setattr(cell, 'output', output) 
def if_similar(a, b, tolerance=0.05):
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
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(l)
    output_y = None
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    print(if_similar(quantity.get_output()[0], ops.norm(output_y)))
    handle.remove()
def test_conv():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(conv)
    handle = conv.register_forward_hook(forward_hook_fn)
    output_y = None
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    print(if_similar(quantity.get_output()[0], ops.norm(output_y)))
    handle.remove()
def test_default():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    output_y = None    
    handle = bn.register_forward_hook(forward_hook_fn)    
    loss_fn = nn.MSELoss()  # 均方误差
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    # 6. 进行前向和反向传播
    model_with_loss = nn.WithLossCell(bn, loss_fn)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for epoch in range(1):  # 单轮训练示例
        loss = train_step(x_default, target)
        output_y = bn(x_default)
        quantity.track(epoch)
        if output_y is not None:
            print(if_similar(quantity.get_output()[0], ops.norm(output_y)))
    handle.remove()

test_linear()
test_conv()
test_default()