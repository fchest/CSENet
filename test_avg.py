import torch
import torch.nn as nn
import torch.nn.init as init
import os
import sys
# from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT 

from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations
            
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Conv_regression(nn.Module):
    def __init__(self,channel, length):  #(64,7999)
        super(Conv_regression, self).__init__()
        self.conv1 = nn.Conv1d(channel,512,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu= nn.ReLU()
        self.avgpool1=nn.AdaptiveAvgPool1d(int(length/3))
        self.avgpool2 = nn.AdaptiveAvgPool1d( int(length / 9))
        self.avgpool3 = nn.AdaptiveAvgPool1d( int(length / 27))
        self.linear1 = nn.Linear(int(length / 27),1)
        self.linear2 = nn.Linear(128, 1)


    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        # print('y1: ', y.size())
        y = self.avgpool1(y)
        # print('y2: ', y.size())
        y = self.relu(self.bn2(self.conv2(y)))
        # print('y3: ', y.size())
        y = self.avgpool2(y)
        # print('y4: ', y.size())
        y = self.relu(self.bn3(self.conv3(y)))
        # print('y5: ', y.size())
        y = self.avgpool3(y)
        # print('y6: ', y.size())
        y = self.linear1(self.relu(y))
        y = torch.squeeze(y)
        y = self.linear2(self.relu(y))
        return y
        
class Conv_regression_selfattention(nn.Module):
    def __init__(self,channel):  #(64,7999)
        super(Conv_regression_selfattention, self).__init__()
        self.conv5 = nn.Conv2d(channel, 256, kernel_size=(4, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.activation = nn.ReLU()

        self.attention = SelfAttention(256)
        
        self.fc = nn.Linear(256 * 2, 128)
        self.fc_mu = nn.Linear(128, 1)


    def forward(self, x):
        # print('x1: ', x.size())
        x = self.conv5(x)
        # print('x2: ', x.size())
        x = self.activation(self.bn5(x)).squeeze(2)
        # print('x3: ', x.size())

        stats = self.attention(x.permute(0, 2, 1).contiguous())
        # print('stats: ', stats.size())

        feat = self.fc(stats)
        # print('x4: ', feat.size())

        mu = self.fc_mu(feat)
        # print('x5: ', mu.size())
        return mu
        
        
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out
        
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
# RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  # '28': [[3, 4, 6, 3], PreActBlock],
                  # '34': [[3, 4, 6, 3], PreActBlock],
                  # '50': [[3, 4, 6, 3], PreActBottleneck],
                  # '101': [[3, 4, 23, 3], PreActBottleneck]
                  # }
RESNET_CONFIGS = {'18': [[2, 2, 2, 2], SEBasicBlock]}

class DCCRN_avg(nn.Module):

    def __init__(
                    self, 
                    rnn_layers=2,
                    rnn_units=128,
                    win_len=400,
                    win_inc=100, 
                    fft_len=512,
                    win_type='hanning',
                    use_clstm=False,
                    use_cbn = False,
                    kernel_size=5,
                    kernel_num=[16,32,64,128,256,256],
                    resnet_type='18'
                ):
        ''' 
            
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(DCCRN_avg, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        input_dim = win_len
        
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        #self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        #self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2]+kernel_num 
        self.use_clstm = use_clstm
        
        #bidirectional=True
        bidirectional=False
        fac = 2 if bidirectional else 1 


        fix=True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        
        # resnet
        self.in_planes = 16
        enc_dim = 256
        layers, block = RESNET_CONFIGS[resnet_type]
        self._norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(6, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(6, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    

    def forward(self, inputs, weighted):
        # print('input: ', inputs.size())
        specs = self.stft(inputs)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]
        # print('cspecs: ', cspecs.size())
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        ''' 

        # print('cspecs: ', cspecs.size())
        x = self.conv1(cspecs)
        # print('x2: ', x.size())
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        # print('x3: ', x.size())
        x = self.layer2(x)
        # print('x4: ', x.size())
        x = self.layer3(x)
        # print('x5: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        x = self.bn5(self.conv5(x))
        # print('conv5: ', x.size())
        x = self.bn6(self.conv6(x))
        x = self.activation(x).squeeze(2)
        # print('x8: ', x.size())

        stats = self.attention(x.permute(0, 2, 1).contiguous())
        # print('stats: ', stats.size())
        
        noise = 1e-5*torch.randn(weighted.size())
        if inputs.is_cuda:
            noise = noise.to(inputs.device)
        avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
        stats = torch.cat((avg_repr,std_repr),1)

        feat = self.fc(stats)

        mu = self.fc_mu(feat)
        
        
        # y=self.Conv_regression_selfattention(out)
        return mu

