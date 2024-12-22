import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from MMonitor.quantity.multistep import *   
from MMonitor.utils.schedules import linear

def is_seasonable(a,model,name):
    print(f"{model}的{name}指标的计算值{a}")
    if a < 0:
        return False
    if a > 1:
        return False
    return True

# 创建BatchNorm层
l = nn.BatchNorm2d(3)
# 使用numpy生成随机数据，然后转换为mindspore tensor
x = Tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))

quantity_l = MeanTID(l, linear(2, 0))

i = 0
y = l(x)
setattr(l, 'input', x)
quantity_l.track(i)
model = 'mindspore_bn'
name = 'mean_tid'
print(is_seasonable(quantity_l.get_output()[0],model,name))
