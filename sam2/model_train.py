import copy
from datetime import time
from random import shuffle
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import  pandas as pd
def train_val_data_process():
    # train_data = FashionMNIST(root="./data",
    #                           train=True,
    #                           transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
    #                           download=True)
    #
    # train_data,val_data = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_dataloader_ir = r"sam2/data/MSRS/train/ir"
    train_dataloader_vis = r"sam2/data/MSRS/train/vis"
    val_dataloader_ir = r"sam2/data/MSRS/test/ir"
    val_dataloader_vis = r"sam2/data/MSRS/test/vis"
    train_labels = r"sam2/data/MSRS/train/Segmentation_labels"
    val_labels = r"sam2/data/MSRS/test/Segmentation_labels"

    return train_dataloader_ir, train_dataloader_vis, val_dataloader_ir, val_dataloader_vis, train_labels, val_labels

def train_model_process(model,train_dataloader_ir, train_dataloader_vis, val_dataloader_ir, val_dataloader_vis,num_epochs):
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备当中
    model = model.to(device)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精准度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    since = time.time()


    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)


        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0.0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0.0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 取数据
        for step,(b_x,b_y) in enumerate(zip(train_dataloader_ir, train_dataloader_vis)):
            # 将特征放入设备
            b_x, b_y = b_x.to(device), b_y.to(device)
            # 将模型设置为训练模式
            model.train()

            output = model(b_x,b_y)

            pred = torch.argmax(output, dim=1)

            # 计算每一个batch损失函数
            loss = criterion(output, b_y)
            train_loss += loss.item()

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 计算损失函数进行累加
            train_loss += loss.item()* b_x.size(0)
            # 如果预测正确，则准确度+1
            train_corrects += torch.sum(pred == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step,(b_x,b_y) in enumerate(zip(train_dataloader_ir, train_dataloader_vis)):
            # 将特征放入设备
            b_x, b_y = b_x.to(device), b_y.to(device)
            # 将模型设置为验证模式
            model.eval()
            # 前向传播，输入一个batch，输出为batch对应的预测
            output = model(b_x,b_y)
            # 查找每一行最大值对应的行标
            pred = torch.argmax(output, dim=1)
            # 计算损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度+1
            val_corrects += torch.sum(pred == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)
        #计算并保存每一次迭代的loss和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)

        print('{} Train Loss:{:.4f} Train ACC: {:.4F}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} VAL Loss:{:.4f} Val ACC: {:.4F}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        # 寻找最高准确度权重
        if val_loss_all[-1] > best_acc:
            best_acc = val_loss_all[-1]
            # 保存当前参数
            best_model_wts = copy.deepcopy(model.state_dict())
        #计算耗时
        time_use = time.time()-since
        print("训练耗时：{:.0f}m{:.0f}s".format(time_use//60,time_use%60))

        # 选择最优参数
        # 加载最高准确率下的模型参数
        model.load_state_dict(best_model_wts)
        torch.save(model.load_state_dict(best_model_wts),r"sam2/best_model.pth")


        train_process = pd.DataFrame(data = {"epoch":range(num_epochs),
                                             "train_loss_all":train_loss_all,
                                             "train_acc_all":train_acc_all,
                                             "val_loss_all":val_loss_all,
                                             "val_acc_all":val_acc_all,
                                             })
        return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label = "Train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'ro-', label="val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="Train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'ro-', label="val acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")