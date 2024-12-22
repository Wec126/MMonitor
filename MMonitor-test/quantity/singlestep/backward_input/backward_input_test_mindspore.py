import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
from MMonitor.quantity.singlestep import *
from mindspore import Tensor
import numpy as np
# 添加更完整的随机种子设置
import random
random.seed(42)
np.random.seed(42)
ms.set_seed(42)
def backward_hook_fn(cell_id,grad_input,grad_output):
    global input_grad
    input_grad = grad_output[0] #（1，10）
def if_similar(a, b, model,name,tolerance=0.05):
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
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputMean(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_mean = ops.reduce_mean(input_grads)
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    model = 'mindspore_linear'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean,model,name))
    handle.remove()
def test_conv_mean():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputMean(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_mean = ops.reduce_mean(input_grads)

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    model = 'mindspore_conv'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_mean():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputMean(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_mean = ops.reduce_mean(input_grads)
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    model = 'mindspore_bn'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean,model,name))
    handle.remove()
def test_linear_norm():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputNorm(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_norm = ops.norm(ops.flatten(input_grads))
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], input_grads_norm,model,name))
    handle.remove()
def test_conv_norm():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputNorm(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_norm = ops.norm(ops.flatten(input_grads))

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], input_grads_norm.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_norm():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputNorm(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_norm = ops.norm(ops.flatten(input_grads))
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_bn'
    print(if_similar(quantity.get_output()[0], input_grads_norm,model,name))
    handle.remove()
def test_linear_std():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputStd(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_std = ops.std(input_grads)
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], input_grads_std,model,name))
    handle.remove()
def test_conv_std():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputStd(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_std = ops.std(input_grads)

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], input_grads_std.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_std():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputStd(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_std = ops.std(input_grads)
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_bn'
    print(if_similar(quantity.get_output()[0], input_grads_std,model,name))
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
