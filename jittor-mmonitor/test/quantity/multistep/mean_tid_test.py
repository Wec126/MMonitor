
import jittor as jt
from jittor import nn
from MMonitor.quantity.multistep import *   
from MMonitor.utils.schedules import linear
def is_seasonable(a):
    print(f"计算值{a.item()}")
    if a < 0 :
        return False
    if a > 1 :
        return False
    return True

l = nn.BatchNorm(3)
x = jt.randn((4, 3, 3))  


quantity_l = MeanTID(l, linear(2, 0))


for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)


i = 0
y = l(x)
quantity_l.track(i)

print(is_seasonable(quantity_l.get_output()[0]))
