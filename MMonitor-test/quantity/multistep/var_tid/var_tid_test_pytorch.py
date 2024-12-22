import torch
import torch.nn as nn
import numpy as np
from MMonitor.quantity.multistep import *   
from MMonitor.utils.schedules import linear

def is_seasonable(a,model,name):
    print(f"{model}的{name}指标的计算值{a.item()}")
    if a < 0:
        return False
    if a > 1:
        return False
    return True
# 定义前向钩子来保存输入
def hook(module, input, output):
    module.last_input = input[0]  # input是一个元组，我们取第一个元素
# 创建BatchNorm层
l = nn.BatchNorm2d(3)
x = torch.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
quantity_l = VarTID(l, linear(2, 0))
for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)
i = 0
y = l(x)
# 不需要手动设置forward_input
quantity_l.track(i)
model = 'pytorch_bn'
name = 'var_tid'
print(is_seasonable(quantity_l.get_output()[0],model,name))
