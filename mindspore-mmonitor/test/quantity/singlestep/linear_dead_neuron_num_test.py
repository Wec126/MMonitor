import mindspore as ms
from mindspore import nn, ops
import numpy as np
from MMonitor.quantity.singlestep import *

def if_similar(a,b):
    print(f"当前计算指标为{a}")
    print(f"预期指标为{b}")
    return a == b
def test_linear():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = LinearDeadNeuronNum(l)
    
    def forward_fn(x):
        output = l(x)
        l.output = output
        return output
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    def train_step(x):
        y, grads = grad_fn(x)
        optimizer(grads)
        return y
    
    i = 0
    y = train_step(x_linear)
    setattr(l, 'output', y)
    quantity.track(i)
    # 手动计算死亡神经元
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx]) < 1e-6):  # 设置阈值
            dead_count += 1
    print(if_similar(quantity.get_output()[0],dead_count))

    
def test_conv():
    cov = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = LinearDeadNeuronNum(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    def train_step(x):
        y, grads = grad_fn(x)
        optimizer(grads)
        return y
    
    i = 0
    y = train_step(x_conv)
    setattr(cov, 'output', y)
    quantity.track(i)
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx]) < 1e-6):  # 设置阈值
            dead_count += 1
    print(if_similar(quantity.get_output()[0],dead_count))

def test_default():
    # 使用Sigmoid层
    x_default = ms.Tensor(np.random.randn(32, 2), dtype=ms.float32)
    sigmoid = nn.Sigmoid()
    quantity = LinearDeadNeuronNum(sigmoid)
    
    def forward_fn(x):
        return sigmoid(x)
    
    i = 0
    y = forward_fn(x_default)
    setattr(sigmoid, 'output', y)
    quantity.track(i)
    
    output = y
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if np.all(np.abs(output[:, neuron_idx].asnumpy()) < 1e-6):  
            dead_count += 1
    print(if_similar(quantity.get_output()[0], dead_count))

test_linear()
test_conv()
test_default()