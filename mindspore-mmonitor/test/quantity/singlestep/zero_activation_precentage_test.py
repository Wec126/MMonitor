import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from MMonitor.quantity.singlestep import *
import numpy as np

def if_similar(a, b):
    print(f"当前计算指标为{a}")
    print(f"预期指标为{b}")
    return a == b
def test_linear():
    # 初始化网络和数据
    l = nn.Dense(2, 3)  # mindspore使用Dense替代Linear
    x_linear = Tensor(np.random.randn(4, 2).astype(np.float32))
    
    # 定义优化器
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    
    # 设置量化器
    quantity = ZeroActivationPrecentage(l)
    i = 0
    y = l(x_linear)
    loss = P.ReduceSum()(y)
    
    # 执行反向传播
    grad_fn = ms.value_and_grad(lambda x: P.ReduceSum()(l(x)), None, optimizer.parameters)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    setattr(l, 'output', y)
    quantity.track(i)
    zero_count = (y.asnumpy() == 0).sum()
    total_elements = y.asnumpy().size
    expected_percentage = zero_count / total_elements
    actual_percentage = quantity.get_output()[0]
    print(if_similar(actual_percentage, expected_percentage))

def test_conv():
    # 初始化卷积网络
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = Tensor(np.random.randn(4, 1, 3, 3).astype(np.float32))
    
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ZeroActivationPrecentage(conv)
    
    i = 0
    y = conv(x_conv)
    loss = P.ReduceSum()(y)
    
    grad_fn = ms.value_and_grad(lambda x: P.ReduceSum()(conv(x)), None, optimizer.parameters)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    setattr(conv, 'output', y)
    quantity.track(i)
    zero_count = (y.asnumpy() == 0).sum()
    total_elements = y.asnumpy().size
    expected_percentage = zero_count / total_elements
    actual_percentage = quantity.get_output()[0]
    print(if_similar(actual_percentage, expected_percentage))

def test_default():
    x_default = Tensor(np.array([[[[0.5, 1.2]], 
                                [[2.3, -1.4]], 
                                [[1.7, 0.8]]]], dtype=np.float32))
    
    bn = nn.BatchNorm2d(3)  # 将通道数改为3以匹配输入tensor的channel维度
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ZeroActivationPrecentage(bn)
    
    i = 0
    y = bn(x_default)
    loss = P.ReduceSum()(y)
    
    grad_fn = ms.value_and_grad(lambda x: P.ReduceSum()(bn(x)), None, optimizer.parameters)
    loss_value, grads = grad_fn(x_default)
    optimizer(grads)
    setattr(bn, 'output', y)
    quantity.track(i)
    zero_count = (y.asnumpy() == 0).sum()
    total_elements = y.asnumpy().size
    expected_percentage = zero_count / total_elements
    actual_percentage = quantity.get_output()[0]
    print(if_similar(actual_percentage, expected_percentage))

if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)  # 设置运行模式
    test_linear()
    test_conv()
    test_default()