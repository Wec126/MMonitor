import jittor as jt
import jittor.nn as nn
from model.model import Model
from MMonitor.extensions.forward_extension import ForwardInputEigOfCovExtension

def prepare_data(length=100, w=224, h=224):
    x = jt.random((length, 3, w, h))
    x.requires_grad=True
    return x

def test_model():
    model = Model()
    forward_input_eig_of_cov_extension = ForwardInputEigOfCovExtension()
    x = prepare_data()
    model.register_forward_hook(forward_input_eig_of_cov_extension)
    y = model(x)

    print(f"The median value obtained by the current extension is {model.input_eig_data.shape}")

def test_linear():
    model = Model()
    forward_input_eig_of_cov_extension = ForwardInputEigOfCovExtension()
    x = prepare_data()
    l = model.l1
    l.register_forward_hook(forward_input_eig_of_cov_extension)
    y = model(x)
    print(f"The median value obtained by the current extension is {l.input_eig_data.shape}")  # 与输入维度相同
def test_Conv2d():
    model = Model()
    forward_input_eig_of_cov_extension = ForwardInputEigOfCovExtension()
    x = prepare_data()
    conv2d = model.conv1
    conv2d.register_forward_hook(forward_input_eig_of_cov_extension)
    y = model(x)
    print(f"The median value obtained by the current extension is {conv2d.input_eig_data.shape}")
# Run tests
test_model()
test_Conv2d()
test_linear()
