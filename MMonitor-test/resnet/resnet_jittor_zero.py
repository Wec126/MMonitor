import jittor as jt
import wandb
from jittor import nn, Module
from jittor.dataset import Dataset, DataLoader
from jittor.models import resnet18
import numpy as np
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization
# 配置 Jittor 运行环境
jt.flags.use_cuda = jt.has_cuda
from jittor.dataset import CIFAR10
# 自定义 CIFAR-10 数据集加载
class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        data_dir = '/data/wlc/dataset/mmonitor'
        cifar10 = CIFAR10(root=data_dir, train=train, download=True)
        self.data = cifar10.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0  # HWC -> CHW, normalize to [0, 1]
        self.labels = np.array(cifar10.targets).astype(np.int32)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.data)
def prepare_config():
    config_mmonitor_resnet18 = {
        nn.BatchNorm: ['ZeroActivationPrecentage','LinearDeadNeuronNum']
    }
    return config_mmonitor_resnet18
# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch,wandb,mmonitor,vis_model):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 转换为 Jittor 张量
        images = jt.array(images)
        labels = jt.array(labels)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        optimizer.step(loss)
        
        # 统计信息
        total_loss += loss.item()
        predicted = jt.argmax(outputs, dim=1)
        correct += (predicted[0] == labels).sum().item()
        total += labels.shape[0]

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    mmonitor.track(epoch)
    logs = vis_model.show(epoch)
    wandb.log(logs)
    print(f"Epoch [{epoch}] Training Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100.0 * correct / total:.2f}%")

# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with jt.no_grad():
        for images, labels in test_loader:
            images = jt.array(images)
            labels = jt.array(labels)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = jt.argmax(outputs, dim=1)
            correct += (predicted[0] == labels).sum().item()
            total += labels.shape[0]
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {100.0 * correct / total:.2f}%")

# 主程序
def main():
    # 加载 CIFAR-10 数据集
    wandb.init(project="resnet_noise_jittor")
    batch_size = 100
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 初始化一个新的AIM运行
    # run = aim.Run()
    # run.name='resnet_jittor_backward_input'
    # 初始化模型、损失函数和优化器
    model = resnet18(num_classes=10)  # CIFAR-10 有 10 个类别
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 训练与测试
    epochs = 100
    config_mmonitor_resnet = prepare_config()

    monitor_resnet = Monitor(model, config_mmonitor_resnet)
    vis_model = Visualization(monitor_resnet, project=config_mmonitor_resnet.keys(),
                              name=config_mmonitor_resnet.values())
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch,wandb,monitor_resnet,vis_model)
        test(model, test_loader, criterion)

if __name__ == "__main__":
    main()
