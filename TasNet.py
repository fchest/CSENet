# wujian@2018
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchsummary import summary
from dc_crn import DCCRN

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles
def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))
def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))
def foo_conv_tas_net():
    x = th.rand(4, 1000)
    nnet = ConvTasNet(norm="cLN", causal=False)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x
class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
class Conv_regression_up(nn.Module):
    def __init__(self,channel,num,len):
        super(Conv_regression_up, self).__init__()
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

    def forward(self, x):
        x = x.float()
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.avgpool1(y)
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.avgpool2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = torch.squeeze(y)
        y = self.linear2(y)
        return y
class Conv_regression(nn.Module):
    def __init__(self,channel, length):  #(64,7999)
        super(Conv_regression, self).__init__()
        self.conv1 = nn.Conv2d(channel,128,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu= nn.ReLU()
        self.avgpool1=nn.AdaptiveAvgPool2d(int(length/4))
        self.avgpool2 = nn.AdaptiveAvgPool2d( int(length / 16))
        self.linear1 = nn.Linear(int(length / 16),1)
        self.linear2 = nn.Linear(256, 1)


    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.avgpool1(y)
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.avgpool2(y)
        y = self.linear1(self.relu(y))
        y = torch.squeeze(y)
        y = self.linear2(self.relu(y))
        return y

def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)
class Conv1D(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class Conv1DBlock(nn.Module):

    def __init__(self,in_channels=256,conv_channels=512,kernel_size=3,dilation=1,norm="cLN",causal=False):
        super(Conv1DBlock, self).__init__()
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.ReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(conv_channels,conv_channels,kernel_size,padding=dconv_pad, dilation=dilation,bias=True)
        self.prelu2 = nn.ReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        self.lnorm3 = build_norm(norm, in_channels)
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.prelu1(self.lnorm1(y))
        y = self.dconv(y)
        y = self.prelu2(self.lnorm2(y))
        y = self.sconv(y)
        #y = self.prelu2(self.lnorm3(y))
        x = x + y
        return x


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1,-2)).transpose(-1,-2)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class ConvTasNet(nn.Module):
    def __init__(self,
                 L=10,  #初始卷积核
                 N=64,  #初始卷积通道数
                 X=8,
                 R=4,
                 B=64,  #1*1卷积通道数
                 H=128, #block卷积通道数
                 P=3,   #block 卷积核
                 norm="BN",
                 causal=False):
        super(ConvTasNet, self).__init__()
        self.encoder_1d = DCCRN(rnn_units=256,use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
        #self.decoder_1d = ConvTrans1D(N, 1, kernel_size=L, stride=L // 2, bias=True)
        # self.conv_regression_up = Conv_regression_up(1,64, 7999)
        self.conv_regression = Conv_regression(64, 7999)
        self.conv_regression3s = Conv_regression(64, 4799)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.eca = eca_layer()
        self.linear = nn.Linear(128,1)

    def _build_blocks(self, num_blocks, **block_kwargs):

        blocks = [Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        repeats = [self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)]
        return nn.Sequential(*repeats)

    def forward(self, x):
        print('input: ', x.size())
        w = F.relu(self.encoder_1d(x))
        print('encoder dccrn: ', w.size())
        y = self.proj(self.ln(w))
        y = self.repeats(y)
        y = self.eca(y)
        y=self.conv_regression3s(y)
        return y

# if __name__ == "__main__":
#
#
#
#     a = torch.tensor([[[1, 2, 3]]])
#     # 新建data3.pkl文件准备写入
#     data_output = open('data3.pkl', 'wb')
#     # 把a写入data.pkl文件里
#     pickle.dump(a, data_output)
#     #关闭写入
#     data_output.close()
#     pathname = "data3.pkl"
#     fp = open(pathname, "rb")
#     x = pickle.load(fp)  # x表示当前的矩阵
#     sns.set()
#     ax = sns.heatmap(x, cmap="rainbow")  # cmap是热力图颜色的参数
#     plt.show()

