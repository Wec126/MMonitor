import jittor as jt
import numpy as np
from jittor import nn
from model.model import Model
from MMonitor.extensions.backward_extension import *


def prepare_data(len=100, w=224, h=224, class_num=5):
    x = jt.randn((len, 3, w, h), requires_grad=True)
    y = jt.randint(0, class_num, (len,))
    return (x, y)

    
def test_model():
    model = Model()
    optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
    (x, y) = prepare_data()
    backward_input_extension = BackwardOutputExtension()
    model.register_backward_hook(backward_input_extension)
    y_hat = model(x)
    loss = nn.cross_entropy_loss(y_hat, y)
    # 反向传播
    optimizer.set_input_into_param_group((x, y))
    optimizer.step(loss)
    print(f"The shape of the median value obtained by the current extension is{model.output_grad.shape}") # 与输入维度相同
def test_linear():
    model = Model()
    backward_input_extension = BackwardOutputExtension()
    (x,y) = prepare_data()
    l = model.l1
    optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
    l.register_backward_hook(backward_input_extension)
    y_hat = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_hat, y)
    optimizer.set_input_into_param_group((x,y))
    optimizer.step(loss)
    print(f"The shape of median value obtained by the current extension is{l.output_grad.shape}") # 与输入维度相同
def test_Conv2d():
    model = Model()
    backward_input_extension = BackwardOutputExtension()
    (x,y) = prepare_data()
    conv2d = model.conv1
    optimizer = jt.optim.SGD(model.parameters(),lr=0.01)
    conv2d.register_backward_hook(backward_input_extension)
    y_hat = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_hat, y)
    optimizer.set_input_into_param_group((x,y))
    optimizer.step(loss)
    print(f"The shape of median value obtained by the current extension is{conv2d.output_grad.shape}") # 与输入维度相同
test_model()
test_linear()
test_Conv2d()
