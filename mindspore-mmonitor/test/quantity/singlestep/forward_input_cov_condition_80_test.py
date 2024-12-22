import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, Zero, Constant, initializer
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
    ms.set_seed(seed)

def test_linear():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition80(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0], 1))

def test_conv():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovCondition80(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0], 1))

def test_default():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition80(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    print(is_similar(quantity.get_output()[0], 1))

# 运行测试
test_linear()
test_conv()
test_default()
