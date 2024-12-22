import torch
import torch.nn as nn
import numpy as np
from MMonitor.quantity.multistep import *   
from MMonitor.utils.schedules import linear

def is_reasonable(value,model,name,threshold=1e-3):
    """判断参数跳跃值是否在合理范围内
    Args:
        value: 计算得到的参数跳跃值
        threshold: 阈值，默认为0.001
    Returns:
        bool: 是否合理
    """
    print(f"{model}的{name}指标的计算值{value}")
    return abs(value) < threshold
# 定义前向钩子来保存输入
def hook(module, input, output):
    module.last_input = input[0]  # input是一个元组，我们取第一个元素
# 创建BatchNorm层
l = nn.BatchNorm2d(3)
x = torch.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
quantity_l = WeightParamJump(l, linear(2, 0))
i = 0
y = l(x)
# 不需要手动设置forward_input
quantity_l.track(i)
model = 'pytorch_bn'
name = 'weight_param_jump'
print(is_reasonable(quantity_l.get_output()[0],model,name))
