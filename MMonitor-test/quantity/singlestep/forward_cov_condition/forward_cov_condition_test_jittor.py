import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
import numpy as np
def is_similar(a, b,model,name,tolerance=0.1):
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) < tolerance
def is_similar_for_stable_rank(a, b,model,name,tolerance=0.1):
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) > 0
# 在文件开头添加
def setup_seed(seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
def test_linear_condition_20():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_20():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_20():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_50():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_50():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_50():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_80():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_80():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_conv'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_80():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_max_eig():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_max_eig():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_conv'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_max_eig():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_conv_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0],1,model,name))

def test_default_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))
test_linear_condition()
test_conv_condition()
test_default_condition()
test_linear_condition_20()
test_conv_condition_20()
test_default_condition_20()
test_linear_condition_50()
test_conv_condition_50()
test_default_condition_50()
test_linear_condition_80()
test_conv_condition_80()
test_default_condition_80()
test_linear_max_eig()
test_conv_max_eig()
test_default_max_eig()
test_linear_cov_stable_rank()
test_conv_cov_stable_rank()
test_default_cov_stable_rank()

