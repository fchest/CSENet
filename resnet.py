# -*- coding: utf-8 -*-
import logging
import torch.nn as nn
import torch
from torchsummary import summary

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(BasicBlock, self).__init__()
        if(in_channel==out_channel):
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.downsample = None
        else:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=5, stride=4, padding=2)
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel,out_channel,kernel_size=1, stride=4, padding=0),
                nn.BatchNorm1d(out_channel)
            )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        if self.downsample is not None:
            res = self.downsample(x)
            out += res
        out = self.relu(out)
        return out

class BasicBlock_avg(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(BasicBlock_avg, self).__init__()
        if(in_channel==out_channel):
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.downsample = None
        else:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=5, stride=4, padding=1)
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                nn.AvgPool1d(4),
                nn.BatchNorm1d(out_channel)
            )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        out = self.relu(out)
        return out

class BasicBlock_max(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(BasicBlock_max, self).__init__()
        if(in_channel==out_channel):
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.downsample = None
        else:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=5, stride=4, padding=1)
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                nn.MaxPool1d(4),
                nn.BatchNorm1d(out_channel)
            )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        out = self.relu(out)
        return out

class Conv_regression(nn.Module):
    def __init__(self,channel,num,len):  #(1,100,240)
        super(Conv_regression, self).__init__()
        self.conv1 = nn.Conv2d(channel,32,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu= nn.ReLU()
        self.avgpool1=nn.AdaptiveAvgPool2d((int(num/2),int(len/2)))
        self.avgpool2 = nn.AdaptiveAvgPool2d((int(num / 4), int(len / 4)))
        self.flatten = nn.Flatten(2)
        self.linear1 = nn.Linear(int(num/4)*int(len/4),1)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x): #(3,498,40)
        x = x.float()
        y = self.relu(self.bn1(self.conv1(x)))
        #y = self.dropout(y)
        y = self.avgpool1(y)
        y = self.relu(self.bn2(self.conv2(y)))
        #y = self.dropout(y)
        y = self.avgpool2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = torch.squeeze(y)
        y = self.linear2(y)
        return y

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv1d(1,200,kernel_size=300, stride=200,padding=100)
        self.block = BasicBlock(1,32)
        self.block2 = BasicBlock(32, 32)
        self.block3 = BasicBlock(32, 64)
        self.block4 = BasicBlock(64, 64)
        self.block5 = BasicBlock(64, 128)
        self.block6 = BasicBlock(128, 128)
        self.block7 = BasicBlock(128, 256)
        self.block8 = BasicBlock(256, 256)
        self.conv_regression=Conv_regression(1,128,625)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128,1)

    def forward(self, x):
        y = self.block(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        #y = self.block7(y)
        #y = self.block8(y)
        y = y.unsqueeze(1)
        #y = y.permute(0, 1, 3, 2).contiguous()  # (b,1,100,240)
        y = self.conv_regression(y)
        return y

class Resnet_avg(nn.Module):
    def __init__(self):
        super(Resnet_avg, self).__init__()
        self.conv1 = nn.Conv1d(1,200,kernel_size=300, stride=200,padding=100)
        self.block = BasicBlock_avg(1,32)
        self.block2 = BasicBlock_avg(32, 32)
        self.block3 = BasicBlock_avg(32, 64)
        self.block4 = BasicBlock_avg(64, 64)
        self.block5 = BasicBlock_avg(64, 128)
        self.block6 = BasicBlock_avg(128, 128)
        self.block7 = BasicBlock_avg(128, 256)
        self.block8 = BasicBlock_avg(256, 256)
        self.block9 = BasicBlock_avg(256, 512)
        self.block10 = BasicBlock_avg(512, 512)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512,1)

    def forward(self, x):
        y = self.block(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.block8(y)
        y = self.block9(y)
        y = self.block10(y)
        y = self.avgpooling(y)
        y = self.flatten(y)
        y = self.linear(y)
        return y

class Resnet_max(nn.Module):
    def __init__(self):
        super(Resnet_max, self).__init__()
        self.conv1 = nn.Conv1d(1,200,kernel_size=300, stride=200,padding=100)
        self.block = BasicBlock_max(1,32)
        self.block2 = BasicBlock_max(32, 32)
        self.block3 = BasicBlock_max(32, 64)
        self.block4 = BasicBlock_max(64, 64)
        self.block5 = BasicBlock_max(64, 128)
        self.block6 = BasicBlock_max(128, 128)
        self.block7 = BasicBlock_max(128, 256)
        self.block8 = BasicBlock_max(256, 256)
        self.block9 = BasicBlock_max(256, 512)
        self.block10 = BasicBlock_max(512, 512)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512,1)

    def forward(self, x):
        y = self.block(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.block8(y)
        y = self.block9(y)
        y = self.block10(y)
        y = self.avgpooling(y)
        y = self.flatten(y)
        y = self.linear(y)
        return y

if __name__ == "__main__":
    net=Resnet()
    a=torch.randn(1,1,40000)
    b=net(a)
    #net=net.cuda()
    #summary(net,(1,40000))

