from collections import defaultdict
import jittor as jt
import jittor.nn as nn
from model.model import Model  
from MMonitor.mmonitor.monitor import Monitor 
import wandb
from MMonitor.visualize import Visualization  

def prepare_data(w, h, class_num, length):
    x = jt.randn((length, 3, w, h))  
    y = jt.randint(0, class_num, (length,))  
    return x, y

def prepare_optimizer(model, lr=1e-2):
    return nn.SGD(model.parameters(), lr=lr) 

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.Conv2d: ['ForwardInputCovCondition','ForwardInputCovCondition20','ForwardInputCovCondition50','ForwardInputCovCondition80','ForwardInputCovStableRank','ForwardInputCovMaxEig'],
        nn.BatchNorm2d: ['ForwardInputCovCondition','ForwardInputCovCondition20','ForwardInputCovCondition50','ForwardInputCovCondition80','ForwardInputCovStableRank','ForwardInputCovMaxEig']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()  

if __name__ == '__main__':
    config, config_mmonitor = prepare_config()
    wandb.init(project='mmonitor', name='test_quantity_remove_hooksmodule3', config=config)
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    model = Model(config['w'], config['h'], config['class_num']) 
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()
    
    # 初始化 Monitor 和 Visualization
    monitor = Monitor(model, config_mmonitor)
    vis = Visualization(monitor, project=config_mmonitor.keys(), name=config_mmonitor.values())
    
    for epoch in range(config['epoch']):
        opt.zero_grad() 
        y_hat = model(x)
        loss = loss_fun(y_hat, y)
        
        # 监控和可视化更新
        monitor.track(epoch)
        logs = vis.show(epoch)
        wandb.log(logs)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        opt.step(loss) 
    wandb.finish()
    print(monitor.get_output())
