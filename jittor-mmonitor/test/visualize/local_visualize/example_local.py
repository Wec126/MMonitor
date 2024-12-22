import jittor as jt
import jittor.nn as nn
from model.model import Model
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization, LocalVisualization

def prepare_data(w, h, class_num, length):
    x = jt.random((length, 3, w, h))  # Jittor中使用jt.random
    y = jt.randint(0, class_num, (length,))
    return x, y

def prepare_optimizer(model, lr=1e-2):
    return jt.optim.SGD(model.parameters(), lr=lr)

def show_local(monitor, quantity_name=None):
    project = 'BatchNorm'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: [['MeanTID', 'linear(5,0)'],'ForwardInputSndNorm','VarTID','ForwardInputMean','ForwardInputStd'],
        nn.Conv2d: ['ForwardInputSndNorm','ForwardInputMean','ForwardInputStd'],
        nn.Linear: ['ForwardInputSndNorm','ForwardInputMean','ForwardInputStd']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()

if __name__ == '__main__':
    config, config_mmonitor = prepare_config()
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    model = Model(config['w'], config['h'], config['class_num'])
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()
    
    #######################################
    monitor = Monitor(model, config_mmonitor)
    vis = Visualization(monitor, project=config_mmonitor.keys(), name=config_mmonitor.values())
    #######################################
    for epoch in range(100):
        opt.zero_grad()
        y_hat = model(x)
        loss = loss_fun(y_hat, y)
        ##############################
        monitor.track(epoch)
        vis.show(epoch)
        print(epoch)
        ##############################
        opt.step(loss)
    print(monitor.get_output())
    show_local(monitor)
