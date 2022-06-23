# -*- coding: utf-8 -*-
import logging
import math
import torch.utils.checkpoint as cp
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchsummary import summary

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)
from collections import OrderedDict
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('pool', nn.BatchNorm2d(num_output_features))
        #self.add_module('pool', nn.BatchNorm2d(num_output_features))
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=3, block_config=(3, 6, 12), compression=0.5,
                 num_init_features=6, bn_size=4, drop_rate=0.5,
                 classes_num=1,input_channel=1, small_inputs=False, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_channel, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, classes_num)
        self.relu=nn.ReLU(inplace=True)
        self.adaptive_avg_pool2d=nn.AdaptiveAvgPool2d((1, 1))
        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = self.adaptive_avg_pool2d(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class SelfAttention(nn.Module):

    def __init__(self,len):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(len, len)  # 128, 128
        self.key =   nn.Linear(len, len)
        self.value = nn.Linear(len, len)

    def forward(self, x):
        x = x.to(torch.float32)
        #x=x.permute(0, 2, 1).contiguous()
        q, k, v = self.query(x), self.key(x), self.value(x)
        y=torch.bmm(q, k.permute(0, 2, 1).contiguous())
        beta=F.softmax(y,dim=1)
        y=torch.bmm(beta, v)
        #y = y.permute(0, 2, 1).contiguous()
        return y

class Conv3d_Net(nn.Module):
    def __init__(self,channel,num,len):
        super(Conv3d_Net, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu= nn.ReLU()
        self.maxpool1=nn.AdaptiveAvgPool2d((int(num/2),int(len/2)))
        self.maxpool2 = nn.AdaptiveAvgPool2d((int(num / 4), int(len / 4)))
        self.flatten = nn.Flatten(2)
        self.linear1 = nn.Linear(1240 if num==498 else 740,1)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x): #(3,498,40)
        x = x.float()
        y = self.bn1(x)
        y = self.relu(self.conv1(y))
        y = self.dropout(y)
        y = self.bn2(y)
        y = self.maxpool1(y)
        y = self.relu(self.conv2(y))
        y = self.dropout(y)
        y = self.bn3(y)
        y = self.maxpool2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = torch.squeeze(y)
        y = self.linear2(y)
        return y

class Conv2d_Net(nn.Module):
    def __init__(self,num,len):
        super(Conv2d_Net, self).__init__()
        self.bn1 = nn.BatchNorm2d(num)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1,16,kernel_size=5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2)
        self.relu= nn.ReLU()
        self.maxpool1=nn.AdaptiveMaxPool2d((int(num/4),int(len/4)))
        self.maxpool2 = nn.AdaptiveMaxPool2d((int(num / 16), int(len / 16)))
        self.flatten = nn.Flatten(2)
        self.linear1 = nn.Linear(185, 1)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x): #(498,40)
        x = x.float()
        y = self.bn1(x)
        y = y.unsqueeze(0)
        y = self.relu(self.conv1(y))

        y = self.dropout(y)
        y = self.bn2(y)
        y = self.maxpool1(y)
        y = self.relu(self.conv2(y))
        y = self.dropout(y)
        y = self.bn3(y)
        y = self.maxpool2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = torch.squeeze(y)
        y = self.linear2(y)
        return y

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

class Conv_operation(nn.Module):
    def __init__(self):
        super(Conv_operation, self).__init__()
        self.bn1=nn.BatchNorm1d(1)
        self.con1d=nn.Conv1d(1, 240, kernel_size=240, stride=1, padding=1)
        #self.con2d=nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #self.con2d2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.con1d2 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #y=self.bn1(x)
        y=self.con1d(x)
        y=y.unsqueeze(1)
        y=self.con2d(y)
        y=self.con2d2(y)
        y = y.squeeze(1)
        y = self.con1d2(y)
        return y

class Sample_Net_dense(nn.Module):
    def __init__(self,len):
        super(Sample_Net_dense, self).__init__()
        self.densenet = DenseNet(input_channel=1)
        #self.conv_regression = Conv_regression(1, 100, int(len / 100))
        self.conv_operation = Conv_operation()
        self.len = int(len / 100)
        self.bn = nn.BatchNorm1d(1)
        self.con2d = nn.Conv2d(1, int(len / 100), kernel_size=(1, int(len / 100)), stride=1, padding=0)

    def forward(self, x):
        x = self.bn(x)
        y = x.view(-1, 100, self.len)
        y = y.unsqueeze(1)
        y = self.con2d(y)
        y = y.permute(0, 3, 2, 1)
        # y = self.conv_operation(y)
        y = self.densenet(y)
        return y

class Sample_Net_conv2d(nn.Module):
    def __init__(self,len):
        super(Sample_Net_conv2d, self).__init__()
        self.densenet=DenseNet(input_channel=1)
        self.conv_regression=Conv_regression(1,100,int(len/100))
        self.conv_operation=Conv_operation()
        self.len=int(len/100)
        self.bn=nn.BatchNorm1d(1)
        self.con2d = nn.Conv2d(1,int(len/100),kernel_size=(1,int(len/100)), stride=1, padding=0)

    def forward(self, x):
        x = self.bn(x)
        y = x.view(-1, 100, self.len)
        y = y.unsqueeze(1)
        y = self.con2d(y)
        y = y.permute(0, 3, 2,1)
        y = self.conv_regression(y)
        return y

class Sample_Net_conv1d(nn.Module):
    def __init__(self,sum_len):
        super(Sample_Net_conv1d, self).__init__()
        len = int(sum_len / 100)
        self.densenet=DenseNet(input_channel=1)
        self.conv_regression=Conv_regression(1,100,len)
        self.conv_operation=Conv_operation()
        self.len=int(sum_len/100)
        self.bn=nn.BatchNorm1d(1)
        self.con1d = nn.Conv1d(1,len,kernel_size=len, stride=len)

    def forward(self, x):
        #print(x.size())
        x = self.bn(x)
        y = self.con1d(x)
        y = y.unsqueeze(1)
        y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
        y = self.conv_regression(y)
        return y

class Sample_Net_conv1d2(nn.Module):
    def __init__(self,sum_len):
        super(Sample_Net_conv1d2, self).__init__()
        len = int(sum_len / 100)
        self.densenet=DenseNet(input_channel=1)
        self.conv_regression=Conv_regression(1,100,len)
        self.conv_operation=Conv_operation()
        self.len=int(sum_len/100)
        self.bn=nn.BatchNorm1d(1)
        self.relu=nn.ReLU()
        self.con1d = nn.Conv1d(1,64,kernel_size=len, stride=len)
        self.con1d2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.con1d3 = nn.Conv1d(128, len, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print(x.size())
        x = self.bn(x)
        y = self.relu(self.con1d(x))
        y = self.relu(self.con1d2(y))
        y = self.relu(self.con1d3(y))
        y = y.unsqueeze(1)
        y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
        y = self.conv_regression(y)
        return y

class Sample_Net_conv1d5(nn.Module):
    def __init__(self,sum_len):
        super(Sample_Net_conv1d5, self).__init__()
        self.densenet=DenseNet(input_channel=1)
        self.conv_regression=Conv_regression(1,100,400)
        self.conv_operation=Conv_operation()
        self.len=int(sum_len/100)
        self.bn  = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(100)
        self.relu=nn.ReLU()
        self.conv = nn.Conv1d(1,32,kernel_size=200, stride=100,padding=50)
        self.con1d1 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.con1d2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.con1d3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.con1d4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.con1d5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.con1d6 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.con1d7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.con1d8 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.con1d9 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con1d10 = nn.Conv1d(512, 100, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print(x.size())
        x = self.bn(x)
        y = self.relu(self.bn1(self.conv(x)))
        y = self.relu(self.bn1(self.con1d1(y)))
        y = self.relu(self.bn2(self.con1d2(y)))
        y = self.relu(self.bn2(self.con1d3(y)))
        y = self.relu(self.bn3(self.con1d4(y)))
        y = self.relu(self.bn3(self.con1d5(y)))
        y = self.relu(self.bn4(self.con1d6(y)))
        y = self.relu(self.bn4(self.con1d7(y)))
        y = self.relu(self.bn5(self.con1d8(y)))
        y = self.relu(self.bn5(self.con1d9(y)))
        y = self.relu(self.bn6(self.con1d10(y)))
        y = y.unsqueeze(1)
        #y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
        y = self.conv_regression(y)
        return y

class Sample_Net_conv1d6(nn.Module):
    def __init__(self,sum_len):
        super(Sample_Net_conv1d6, self).__init__()
        self.densenet=DenseNet(input_channel=1)
        self.conv_regression=Conv_regression(1,400,512)
        self.conv_operation=Conv_operation()
        self.len=int(sum_len/100)
        self.bn  = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu=nn.ReLU()
        self.conv = nn.Conv1d(1,32,kernel_size=200, stride=100,padding=50)
        self.con1d1 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.con1d2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.con1d3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.con1d4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.con1d5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.con1d6 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.con1d7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.con1d8 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.con1d9 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print(x.size())
        x = self.bn(x)
        y = self.relu(self.bn1(self.conv(x)))
        y = self.relu(self.bn1(self.con1d1(y)))
        y = self.relu(self.bn2(self.con1d2(y)))
        y = self.relu(self.bn2(self.con1d3(y)))
        y = self.relu(self.bn3(self.con1d4(y)))
        y = self.relu(self.bn3(self.con1d5(y)))
        y = self.relu(self.bn4(self.con1d6(y)))
        y = self.relu(self.bn4(self.con1d7(y)))
        y = self.relu(self.bn5(self.con1d8(y)))
        y = self.relu(self.bn5(self.con1d9(y)))
        y = y.unsqueeze(1)
        y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
        y = self.conv_regression(y)
        return y

# class Sample_Net_conv1d6(nn.Module):
#     def __init__(self,sum_len):
#         super(Sample_Net_conv1d6, self).__init__()
#         len = int(sum_len / 100)
#         self.densenet=DenseNet(input_channel=1)
#         self.conv_regression=Conv_regression(1,100,len)
#         self.conv_operation=Conv_operation()
#         self.len=512
#         self.bn=nn.BatchNorm1d(1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.bn6 = nn.BatchNorm1d(len)
#         self.relu=nn.ReLU()
#         self.con1d = nn.Conv1d(1,32,kernel_size=400, stride=200,padding=100)
#
#         self.con1d1 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.con1d2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.rescon1 = nn.Conv1d(32,64, kernel_size=1, stride=1, padding=0)
#
#         self.con1d3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.con1d4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.rescon2 = nn.Conv1d(64, 256, kernel_size=1, stride=1, padding=0)
#
#         self.con1d5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.con1d6 = nn.Conv1d(512, len, kernel_size=3, stride=1, padding=1)
#         self.rescon3 = nn.Conv1d(256, len, kernel_size=1, stride=1, padding=0)
#         self.dropout=nn.Dropout(0.5)
#         #self.con1d7 = nn.Conv1d(32, len, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         #print(x.size())
#         x = self.bn(x)
#         y = self.relu(self.bn1(self.con1d(x)))
#
#         res = y
#         y = self.relu(self.bn1(self.con1d1(y)))
#         y = self.relu(self.bn2(self.con1d2(y)))
#         res = self.relu((self.rescon1(res)))
#         y=y+res
#
#         res=y
#         y = self.relu(self.bn3(self.con1d3(y)))
#         y = self.relu(self.bn4(self.con1d4(y)))
#         res = self.relu((self.rescon2(res)))
#         y=y+res
#
#         res = y
#         y = self.relu(self.bn5(self.con1d5(y)))
#         y = self.relu(self.bn6(self.con1d6(y)))
#         res = self.relu((self.rescon3(res)))
#         y = y + res
#
#         #y = self.relu((self.con1d7(y)))
#
#         y = y.unsqueeze(1)
#         y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
#         y = self.conv_regression(y)
#         return y

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self,sum_len):
        super(TCN, self).__init__()
        len = int(sum_len / 100)
        self.tcn=TemporalConvNet(1,[32,64,128,64,32,1])
        self.conv_regression=Conv_regression(1,100,len)
        self.con1d = nn.Conv1d(1,512,kernel_size=len, stride=(int)(len/2))


    def forward(self, x):
        y=self.tcn(x)
        y=self.con1d(y)
        y = y.unsqueeze(1)
        y = y.permute(0,1, 3,2).contiguous() #(b,1,100,240)
        y = self.conv_regression(y)
        return y


if __name__ == "__main__":
    net=Conv_regression(1, 64, 7999)
    torch.cuda.set_device(5)
    # a=torch.randn(1,1,40000)
    # b=net(a)
    net=net.cuda()
    summary(net,(1,64,7999))
