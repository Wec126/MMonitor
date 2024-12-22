
import jittor as jt
from jittor import nn
from MMonitor.quantity.singlestep import RankMe

l = nn.Linear(2, 3)
cov = nn.Conv2d(1, 2, 3, 1, 1)


x = jt.randn((4, 2))
x_c = jt.randn((4, 1, 3, 3))


quantity_l = RankMe(l)
quantity_c = RankMe(cov)


for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)

for hook in quantity_c.forward_extensions():
    cov.register_forward_hook(hook)


for i in range(3):
    y = l(x)
    y_c = cov(x_c)
    quantity_l.track(i)
    quantity_c.track(i)


print(quantity_l.get_output())
print(quantity_c.get_output())
