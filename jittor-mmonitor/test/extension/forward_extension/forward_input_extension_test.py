
import jittor as jt
from jittor import nn
from MMonitor.extensions.forward_extension import ForwardInputExtension
import numpy as np
from model.model import Model

# Prepare data function
def prepare_data(len=100, w=224, h=224):
    x = jt.random((len, 3, w, h), dtype=jt.float32)  # Create random tensor
    return x

# Test model function
def test_model():
    model = Model()
    forward_input_extension = ForwardInputExtension()
    x = prepare_data()
    model.register_forward_hook(forward_input_extension)
    y = model(x)

    print(f"The median value obtained by the current extension is {model.input.shape}")
# Test linear layer function
def test_linear():
    model = Model()
    forward_input_extension = ForwardInputExtension()
    x = prepare_data()
    l = model.l1
    l.register_forward_hook(forward_input_extension)
    y = model(x)

    print(f"The median value obtained by the current extension is {l.input.shape}")

# Test Conv2d layer function
def test_Conv2d():
    model = Model()
    forward_input_extension = ForwardInputExtension()
    x = prepare_data()
    conv2d = model.conv1
    conv2d.register_forward_hook(forward_input_extension)
    y = model(x)

    print(f"The median value obtained by the current extension is {conv2d.input.shape}")

# Run the tests
test_model()
test_Conv2d()
test_linear()
