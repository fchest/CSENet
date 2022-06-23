import librosa
import numpy as np
import wave
#from model import *
import torch
import torch.nn as nn
from scipy import signal

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #torch.random.seed(seed)
    torch.backends.cudnn.deterministic = True

#con1d = nn.Conv1d(240, 240, kernel_size=100, stride=1, padding=0)

class Sample_Net_conv(nn.Module):
    def __init__(self):
        super(Sample_Net_conv, self).__init__()
        #self.densenet=DenseNet(input_channel=1)
        #self.conv_regression=Conv_regression(1,100,240)
        #self.conv_operation=Conv_operation()
        #self.len=len
        self.bn=nn.BatchNorm1d(1)
        self.con2d = nn.Conv2d(1,240,kernel_size=(1,240), stride=1, padding=0)

    def forward(self, x):
        x=self.bn(x)
        y=x.view(-1,100,240)
        y = y.unsqueeze(1)
        y=self.con2d(y)
        y = y.squeeze(-1)
        y = y.permute(0,2,1)

        #y = self.conv_operation(y)
        #y = self.conv_regression(y)
        return y

a=torch.randn(1,1,24000)
#print(a)
net=Sample_Net_conv()
b=net(a)
print(b.size())
