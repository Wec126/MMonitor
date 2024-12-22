import mindspore as ms
from mindspore import nn
from MMonitor.quantity.multistep import *   
import numpy as np
def is_seasonable(a):
    print(f"当前计算值{a}")
    if a < 0:
        return False
    if a > 1:
        return False
    return True

l = nn.BatchNorm1d(3)
x = ms.Tensor(np.random.rand(4, 3, 3), dtype=ms.float32)

quantity_l = VarTID(l)


i = 0
y = l(x)  # 前向传播
setattr(l, 'input', x)
quantity_l.track(i)

print(is_seasonable(quantity_l.get_output()[0]))
