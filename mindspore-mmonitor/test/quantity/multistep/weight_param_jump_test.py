import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
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

# 创建LayerNorm层
l = nn.LayerNorm([3])
# 创建随机输入张量
x = Tensor(shape=(4, 9, 3), dtype=ms.float32, init=ms.common.initializer.Normal())

# 初始化监控器
quantity_l = WeightParamJump(l)


# 执行前向传播并跟踪
j = 0
y = l(x)
quantity_l.track(j)

# 获取结果并判断是否合理
result = quantity_l.get_output()[0]
print(is_reasonable(result))
