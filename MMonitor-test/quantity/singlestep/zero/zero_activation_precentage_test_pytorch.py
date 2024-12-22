from MMonitor.quantity.singlestep.zero_activation_precentage import ZeroActivationPrecentage
import torch
from torch import nn as nn

def if_similar(a,b,model,name):
    print(f"{model}的{name}指标的当前计算指标为{a}")
    print(f"{model}的{name}指标的预期指标为{b}")
    if a == b:
        print('True')
    else:
        print('False')
def compute_linear():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = ZeroActivationPrecentage(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    zero_count = (y.detach().numpy() == 0).sum()
    total_elements = y.detach().numpy().size
    expected_percentage = zero_count / total_elements
    model = 'pytorch_linear'
    name = 'zero_activation_precentage'
    if_similar(quantity_l.get_output()[0].item(),expected_percentage,model,name)

def compute_conv():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = ZeroActivationPrecentage(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    quantity_c.track(i)
    zero_count = (y.detach().numpy() == 0).sum()
    total_elements = y.detach().numpy().size
    expected_percentage = zero_count / total_elements
    model = 'pytorch_linear'
    name = 'zero_activation_precentage'
    if_similar(quantity_c.get_output()[0].item(),expected_percentage,model,name)

def compute_default():
    # 创建一个2D的BatchNorm层
    bn = nn.BatchNorm2d(2)
    # 创建一个4D的输入张量：(batch_size, channels, height, width)
    x_default = torch.randn((4, 2, 3, 3))
    
    quantity = ZeroActivationPrecentage(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    
    i = 0
    y = bn(x_default)
    quantity.track(i)
    
    zero_count = (y.detach().numpy() == 0).sum()
    total_elements = y.detach().numpy().size
    expected_percentage = zero_count / total_elements
    model = 'pytorch_linear'
    name = 'zero_activation_precentage'
    if_similar(quantity.get_output()[0].item(), expected_percentage,model,name)

compute_linear()
compute_conv()
compute_default()
    



