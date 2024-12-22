import aim
from aim import Run
import jittor as jt
import jittor.nn as nn
from model.model import Model  
from MMonitor.mmonitor.monitor import Monitor  
from MMonitor.visualize import Visualization  

def prepare_data(w, h, class_num, length):
    x = jt.randn((length, 3, w, h))  
    y = jt.randint(0, class_num, (length,)) 
    return x, y

def prepare_optimizer(model, lr=1e-2):
    return nn.SGD(model.parameters(), lr=lr)  

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor={
        nn.BatchNorm:['ForwardOutputMean','ForwardOutputStd','ForwardOutputGradNorm'],
        nn.Conv2d:['ForwardOutputMean','ForwardOutputStd','ForwardOutputGradNorm'],
        nn.Linear:['ForwardOutputMean','ForwardOutputStd','ForwardOutputGradNorm']
        
    }

    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()  # Jittor 的交叉熵损失函数

if __name__ == '__main__':
    config, config_mmonitor = prepare_config()
    run_model = aim.Run(repo='aim://203.83.235.100:30058', system_tracking_interval=None, log_system_params=False)
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    model = Model(config['w'], config['h'], config['class_num'])  # 假设 Model 已转换为 Jittor
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()
    
    # 初始化 Monitor 和 Visualization
    monitor = Monitor(model, config_mmonitor)
    vis = Visualization(monitor, project=config_mmonitor.keys(), name=config_mmonitor.values())
    
    for epoch in range(config['epoch']):
        y_hat = model(x)
        opt.set_input_into_param_group((x, y))
        loss = loss_fun(y_hat, y)
        # 监控和可视化更新
        opt.step(loss) 
        monitor.track(epoch)
        logs = vis.show(epoch)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        run_model.track(logs,context={'subset':'train'})
    print(monitor.get_output())
