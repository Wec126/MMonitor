import jittor as jt
from jittor import nn
from MMonitor.quantity.multistep.weight_param_jump import WeightParamJump

def is_reasonable(value, threshold=1e-3):
    """判断参数跳跃值是否在合理范围内
    Args:
        value: 计算得到的参数跳跃值
        threshold: 阈值，默认为0.001
    Returns:
        bool: 是否合理
    """
    print(f"参数跳跃值: {value}")
    return abs(value) < threshold

l = nn.LayerNorm(3)
x = jt.randn((4, 9, 3))  


quantity_l = WeightParamJump(l)


for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)


j = 0
y = l(x)
quantity_l.track(j)

result = quantity_l.get_output()[0]
print(is_reasonable(result))
