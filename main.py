# -*- coding: utf-8 -*-
import torch
import logging
import os
from torchsummary import summary
#from resnet import *
from pathlib import Path
from eval import test,test_num
from TasNet import ConvTasNet
from dc_crn import DCCRN
from train import train
from model import *
from dataload import *
classes_num=1
logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

def load_net(net,model_pkl):
    logger.info("load:%s"%model_pkl)
    net.load_state_dict(torch.load("../pkl/"+model_pkl))
    return net
def count_parameters(model):
    parameters_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(parameters_sum)


def train_3s(data_root, save_path):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    
    dev_root = os.path.join(data_root, 'dev')
    val_data=val_data_loader(dev_root, batch_size=64, shuffle=False)
    # val_data=val_data_loader('../audio_3s/dev/',batch_size=64,shuffle=False)
    # net = ConvTasNet(X=4,R=2)  # 模型
    net = DCCRN(rnn_units=256,use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[0])      #GPU配置
    net.to(device)
    train(data_root, net=net, epoch_num=100, trainloader="3s", valloader=val_data,batch_size=64,
          device=device,save_path=save_path,info_num=200,step_size=5, flag=-3)

def test_3s():
    #train_data=train_data_loader('../audio_3s/train/',batch_size=64,shuffle=False)
    test_data=test_data_loader('../audio_3s/test/',batch_size=1,shuffle=False)
    net = ConvTasNet(X=4,R=2)  # 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[0,1])      #GPU配置
    net.to(device)
    # net = load_net(net, "Net_conv1d2_99.pkl")
    # test_num(net=net, testloader=test_data, device=device)
    for i in range(100,0,-1):
        path="3s_1d_"+str(i)+".pkl"
        my_file = Path("../pkl/"+path)
        if my_file.is_file():
            net=load_net(net,path)
            test_num(net=net, testloader=test_data, device=device)

def train_5s():
    val_data=val_data_loader('../audio_5s/dev/',batch_size=8,shuffle=False)
    net = ConvTasNet(X=4,R=2)  # 模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[1])      #GPU配置
    net.to(device)
    train(net=net, epoch_num=1, trainloader="5s", valloader=val_data,batch_size=1,
          device=device,save_path="best_relu",info_num=10,step_size=5)

def test_5s():
    test_data=test_data_loader('../audio_5s/test/',batch_size=1,shuffle=False)
    net = ConvTasNet(X=4,R=2)  # 模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[1])      #GPU配置
    net.to(device)
    best_mse = 100
    best_mae = 100
    for i in range(100, 0, -1):
        path = "best_relu_" + str(i) + ".pkl"
        my_file = Path("../pkl/" + path)
        if my_file.is_file():
            net = load_net(net, path)
            tem_mse,tem_mae = test_num(net=net, testloader=test_data, device=device)
            if (tem_mse < best_mse):
                best_mse = tem_mse
            if (tem_mae < best_mae):
                best_mae = tem_mae
    print(best_mse)
    print(best_mae)


if __name__ == "__main__":
    data_root = '/data3/fancunhang/Depression/audio_good_without_move/AVEC2013_3s/'
    model_save_path = 'exp_0.002/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    train_3s(data_root, model_save_path)


