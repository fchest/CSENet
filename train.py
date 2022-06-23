# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import logging
from dataload import *

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

#训练
def train(data_root, net,epoch_num,trainloader,batch_size,valloader,device=None,save_path=None,info_num=200,step_size=2, flag=-3):
    net.train()
    train_root = os.path.join(data_root, 'train')
    model_save_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
    if trainloader == "3s":
        train_data = train_data_loader(train_root, batch_size=batch_size, shuffle=True,flag=flag)
    best_loss=1000
    criterion = nn.MSELoss()                       #损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.002,weight_decay=0.01)       #优化器
    # optimizer = optim.Adam(net.parameters(), lr=0.0003,weight_decay=0.01)       #优化器
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=-1)
    start = time.time()
    for epoch in range(epoch_num):
        net.train()
        if trainloader=="5s":
            train_data = train_data_loader('../audio_5s/train/', batch_size=batch_size, shuffle=False,flag=3)
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels =data
            # print('dataload input: ', inputs.size())
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            # print('dataload input: ', inputs.size())
            # print('dataload input.float(): ', inputs.float().size())
            outputs = net(inputs.float())
            outputs=outputs.squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % info_num == info_num-1:
                loss_mean=(running_loss/info_num)**0.5
                logger.info('[%d, %5d] %s loss: %.3f' %(epoch + 1, i + 1,timeSince(start), loss_mean))
                running_loss = 0.0
        #scheduler.step()
        torch.save(net.state_dict(), model_save_path + "/model_epoch_" + str(epoch + 1) + ".pkl")
        val_loss = validate(valloader, net,device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), model_save_path + "/best_model_epoch_" + str(epoch + 1) + ".pkl")
    
    logger.info('Best loss: %5f'%best_loss)
    logger.info('Finished Training')


def validate(val_loader, model,device):
    #切换模型为预测模型
    model.eval()
    batch_loss=0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(-1)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels.float())
            batch_loss += loss.item()
            batch_num = i + 1
        mse = (batch_loss / batch_num) ** 0.5
    logger.info("eval mse is: %5f" % mse)
    return mse