
import jittor as jt
from jittor import nn
from MMonitor.quantity.multistep import *   

def is_seasonable(a,model,name):
    print(f"{model}的{name}指标的计算值{a.item()}")
    if a < 0 :
        return False
    if a > 1:
        return False
    return True
l = nn.BatchNorm1d(3)
x = jt.rand((4, 3, 3)) 


quantity_l = VarTID(l)


for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)


i= 0
y = l(x)
quantity_l.track(i)

model = 'jittor_bn'
name = 'var_tid'
print(is_seasonable(quantity_l.get_output()[0],model,name))