import subprocess
import sys
import os
# 安装指定的库
import argparse
import numpy as np
import jittor as jt
import wandb
from data import *
from jittor import nn
from jittor import models
import copy
import datetime
import matplotlib
import seaborn

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from jittor.dataset import Dataset
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization
cifar_10_python_path = '/data/wlc/dataset/mmonitor'
modelPath = '/data/wlc/dataset/mmonitor'
parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--model', default='resnet32', type=str, help='model')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.cifar_10_python_path = cifar_10_python_path
args.json_file_path = '/data/wlc/models/mmonitor'
args.sel_data = list(range(50000))

jt.flags.use_cuda = 1


def build_model():
    model = models.Resnet18(num_classes=args.num_classes)
    return model


def prepare_config():
    config_mmonitor_resnet18 = {
        nn.BatchNorm: ['ForwardInputCovMaxEig','ForwardInputCovStableRank']
    }
    return config_mmonitor_resnet18


def accuracy(output, target):
    batch_size = target.shape[0]
    pred = np.argmax(output, -1)
    res = ((pred == target).astype(float).sum()) / batch_size

    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ce_loss(output, target, reduce=True):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))
    target = target.reshape((-1,))
    target = target.broadcast(output, [1])
    target = target.index(1) == target

    output = output - output.max([1], keepdims=True)
    loss = output.exp().sum(1).log()
    loss = loss - (output * target).sum(1)
    if reduce:
        return loss.mean()
    else:
        return loss


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    for _, (inputs, targets, _) in enumerate(test_loader):
        inputs, targets = jt.array(inputs), jt.array(targets)
        outputs = model(inputs)
        test_loss += nn.cross_entropy_loss(outputs, targets).detach().item()
        predicted = np.argmax(outputs.detach(), -1)
        correct += ((predicted == targets.data).astype(float).sum())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader),
        accuracy))

    return accuracy


def train(train_loader, model, optimizer_model, epoch, epoch_losses, epoch_uncertainty, wandb, monitor_model,
          vis_model):
    current_time = datetime.datetime.now()
    print('\nEpoch: %d, Time: %s' % (epoch, current_time.strftime('%Y-%m-%d %H:%M:%S')))

    train_loss = 0
    prec_train_all = 0.

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        jt.sync_all()
        model.train()

        outputs = model(inputs)
        loss = ce_loss(outputs, targets)
        optimizer_model.step(loss)

        prec_train = accuracy(outputs.data, targets.data)

        train_loss += (loss.item() * outputs.shape[0])
        prec_train_all += (prec_train.item() * outputs.shape[0])

        with jt.no_grad(no_fuse=1):
            unce = (nn.softmax(outputs, dim=1)).max()

            epoch_losses[index] = ce_loss(outputs, targets, reduce=False)
            epoch_uncertainty[index] = unce

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Prec@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader), (train_loss / (batch_idx + 1)),
                      prec_train))
        #打印时间戳
    starttime = datetime.datetime.now()
    print("epoch %d start time: %s", epoch, starttime.strftime('%Y-%m-%d %H:%M:%S'))                  
    monitor_model.track(epoch)
    logs = vis_model.show(epoch)
    wandb.log(logs)
    #打印时间戳
    endtime = datetime.datetime.now()
    print("epoch %d end time: %s", epoch, endtime.strftime('%Y-%m-%d %H:%M:%S'))
    print('epcho beihang tool 记录消耗时间：', (endtime - starttime).seconds)
    return (train_loss / len(train_loader.dataset)), (
                prec_train_all / len(train_loader.dataset)), epoch_losses, epoch_uncertainty


def extract_features(model, dataloader):
    features = []
    labels = []
    feature_extractor = copy.deepcopy(model)
    feature_extractor.linear = nn.Identity()
    with jt.no_grad(no_fuse=1):
        for inputs, targets, _ in dataloader:
            output = feature_extractor(inputs)
            features.append(output.reshape(output.shape[0], -1).numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def get_output(model, dataloader):
    pre = jt.zeros(len(dataloader.dataset), args.num_classes)
    labels = jt.zeros(len(dataloader.dataset)).long()
    feature_extractor = copy.deepcopy(model)
    feature_extractor.eval()
    with jt.no_grad():
        for inputs, targets, index in dataloader:
            output = feature_extractor(inputs)

            pre[index] = output
            labels[index] = targets
    return pre, labels


# load dataset
train_loader = build_dataset(args)
args.num_classes = len(train_loader.classes)

# load model
model = build_model()
optimizer_model = jt.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# 使用wandb进行记录

def main():
    wandb.init(project="resnet_noise_jittor", config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    })
    best_acc = 0
    train_loss_all = []
    prec_train_all = []

    epoch_losses = jt.zeros(len(train_loader.dataset))
    epoch_uncertainty = jt.zeros(len(train_loader.dataset))

    out_file_list = []
    config_mmonitor_resnet = prepare_config()

    monitor_resnet = Monitor(model, config_mmonitor_resnet)
    vis_model = Visualization(monitor_resnet, project=config_mmonitor_resnet.keys(),
                              name=config_mmonitor_resnet.values())
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        train_loss, prec_train, epoch_losses, epoch_uncertainty = train(train_loader, model, optimizer_model, epoch,
                                                                        epoch_losses, epoch_uncertainty, wandb,
                                                                        monitor_resnet, vis_model)

        if (epoch+1) % 5 == 0:
            pre, labels = get_output(model, train_loader)
        print("track metric start time:", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))       
    pre, labels = get_output(model, train_loader)
    wandb.finish()
if __name__ == '__main__':
    main()
