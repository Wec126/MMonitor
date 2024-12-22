import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal
from MMonitor.quantity.singlestep import *
from mindspore import Tensor
import numpy as np
# 计算的是输出的梯度
def if_similar(tensor1, tensor2, tol=1e-5):
    return np.allclose(tensor1.asnumpy(), tensor2.asnumpy(), atol=tol)
def backward_hook_fn(cell, grad_input, grad_output):
    global output_grad
    output_grad = grad_input[0] #获取输出梯度,(1，5)

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
    global output_grad  # 添加全局变量声明
    output_grad = None  # 初始化input_grad
   # 创建一个Dense层
    dense = nn.Dense(10, 5)
    # 准备输入数据
    x = ms.Tensor(np.random.rand(1, 10).astype(np.float32))
    target = ms.Tensor(np.random.rand(1, 5).astype(np.float32))
    quantity = BackwardOutputNorm(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    # 定义损失函数
    loss_fn = nn.MSELoss()
    
    # 前向传播
    logits = dense(x)
    loss = loss_fn(logits, target)
    
    # 手动计算预期的输出梯度
    output_grads = 2 * (logits - target) / logits.size
    output_grads_norm = ops.norm(ops.flatten(output_grads))
    # 确保使用正确的梯度计算方法
    grad_fn = ops.value_and_grad(lambda x: loss_fn(dense(x), target))
    loss_value, _ = grad_fn(x)
    
    # 执行反向传播
    loss.backward()  # 添加retain_graph=True
    
    # 设置捕获的梯度
    if output_grad is not None:
        setattr(dense, 'output_grad', output_grad)
        quantity.track(0)
        print(if_similar(quantity.get_output()[0], output_grads_norm))
    else:
        print("未捕获到梯度")
        
    handle.remove()
def test_conv():
    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    # 定义损失函数
    loss_fn = nn.MSELoss()
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义指标计算类
    quantity = BackwardOutputNorm(conv)
    handle = conv.register_backward_hook(backward_hook_fn)
    # 定义损失函数
    loss_fn = nn.MSELoss()
    
    # 前向传播
    logits = conv(x_conv)
    loss = loss_fn(logits, target)
    
    # 手动计算预期的输出梯度
    output_grads = 2 * (logits - target) / logits.size
    output_grads_norm = ops.norm(ops.flatten(output_grads))
    # 确保使用正确的梯度计算方法
    grad_fn = ops.value_and_grad(lambda x: loss_fn(conv(x), target))
    loss_value, _ = grad_fn(x_conv)
    
    # 执行反向传播
    loss.backward()  # 添加retain_graph=True  
    # 设置捕获的梯度
    if output_grad is not None:
        setattr(conv, 'output_grad', output_grad)
        quantity.track(0)
        print(if_similar(quantity.get_output()[0], output_grads_norm))
    else:
        print("未捕获到梯度")
        
    handle.remove()

def test_default():
    batch_size = 8
    channels = 3
    height = 32
    width = 32
    
    # 定义BatchNorm层
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardOutputNorm(bn)
    handle = bn.register_backward_hook(backward_hook_fn)
    # 定义损失函数
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 前向传播
    logits = bn(x_default)
    loss = loss_fn(logits, target)
    
    # 手动计算预期的输出梯度
    output_grads = 2 * (logits - target) / logits.size
    output_grads_norm = ops.norm(ops.flatten(output_grads))
    # 确保使用正确的梯度计算方法
    grad_fn = ops.value_and_grad(lambda x: loss_fn(bn(x_default), target))
    loss_value, _ = grad_fn(x_default)
    
    # 执行反向传播
    loss.backward()  
    
    # 设置捕获的梯度
    if output_grad is not None:
        setattr(bn, 'output_grad', output_grad)
        quantity.track(0)
        print(if_similar(quantity.get_output()[0], output_grads_norm))
    else:
        print("未捕获到梯度")        
    handle.remove()

test_linear()
test_conv()
test_default() 