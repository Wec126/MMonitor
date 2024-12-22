
from MMonitor.extensions.forward_extension import *
import jittor as jt
from model.model import Model

# Prepare data function
def prepare_data(len=100, w=224, h=224):
    x = jt.random((len, 3, w, h), dtype=jt.float32)  # Create random tensor
    return x

def test_model():
    model = Model()
    forward_output_extension = ForwardOutputExtension()
    x = prepare_data()
    model.register_forward_hook(forward_output_extension)
    y = model(x)

    
    print(f"The median value obtained by the current extension is {model.output.shape}")

def test_linear():
    model = Model()
    forward_output_extension = ForwardOutputExtension()
    x = prepare_data()
    l = model.l1
    l.register_forward_hook(forward_output_extension)
    y = model(x)

    
    print(f"The median value obtained by the current extension is {l.output.shape}")

def test_conv2d():
    model = Model()
    forward_output_extension = ForwardOutputExtension()  # Use output extension
    x = prepare_data()
    conv2d = model.conv1
    conv2d.register_forward_hook(forward_output_extension)
    y = model(x)

    
    print(f"The median value obtained by the current extension is {conv2d.output.shape}")

# Execute tests
test_model()
test_conv2d()
test_linear()
