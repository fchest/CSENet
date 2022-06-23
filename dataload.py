# -*- coding: utf-8 -*-
import os
import random
import wave
import numpy as np
import librosa
import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
from scipy.io import loadmat
from scipy import signal

class Depression_3dmel_random_train(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for j in range(160):
            list_shuffle = []
            labels = os.listdir(root_dir)
            for i in range(len(labels)):
                list = []
                for root, dirs, files in os.walk(root_dir + labels[i]):
                    for file in files:
                        list.append(os.path.join(root, file))
                random.shuffle(list)
                list = list[:10]
                list_shuffle += list
            random.shuffle(list_shuffle)
            self.data_list += list_shuffle
        for k in range(len(self.data_list)):
            self.label_list.append((self.data_list[k].split('/'))[3])

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        waveData=loadmat(self.data_list[index])['value']
        waveData = torch.from_numpy(waveData)
        label = int(self.label_list[index])
        sample = (waveData,label)                # 根据图片和标签创建元组
        return sample
class Depression_3dmel_order_test(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for root, dirs, files in os.walk(root_dir):                    #./clip/train/
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(root_dir+dir):     #./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".mat"):
                            self.data_list.append(os.path.join(root2,file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)                       # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        waveData=loadmat(self.data_list[index])['value']
        name=self.data_list[index].split('/')[-2]
        waveData = torch.from_numpy(waveData)
        label = int(self.label_list[index])
        sample = (waveData,label,str(name))                # 根据图片和标签创建元组
        return sample
class Depression_3dmel_order_dev(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for root, dirs, files in os.walk(root_dir):                    #./clip/train/
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(root_dir+dir):     #./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".mat"):
                            self.data_list.append(os.path.join(root2,file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)                       # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        waveData=loadmat(self.data_list[index])['value']
        #name=self.data_list[index].split('/')[-2]
        waveData = torch.from_numpy(waveData)
        label = int(self.label_list[index])
        sample = (waveData,label)                # 根据图片和标签创建元组
        return sample
def train_data_loader_3d(root='./audio_mat_3dmel_3s/train/',batch_size=64,shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    traindata = Depression_3dmel_random_train(root, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle,num_workers=4)  # 使用DataLoader加载数据
    return trainloader
def test_data_loader_3d(root='./audio_mat_3dmel_3s/test/',batch_size=1,shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testdata = Depression_3dmel_order_test(root, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=shuffle, num_workers=4,drop_last=True)  # 使用DataLoader加载数据
    return testloader
def val_data_loader_3d(root='./audio_mat_3dmel_3s/dev/',batch_size=1,shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testdata = Depression_3dmel_order_dev(root, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=shuffle, num_workers=4,drop_last=True)  # 使用DataLoader加载数据
    return testloader


class Depression_wav_order_train14(Dataset):
    def __init__(self, root_dir,root_dir2, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.root_dir2= root_dir2
        self.transform = transform  # 变换
        # self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list = []
        self.label_list = []
        for root, dirs, files in os.walk(root_dir):  # ./clip/train/
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(root_dir + dir):  # ./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".wav"):
                            self.data_list.append(os.path.join(root2, file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)  # 00
        for root, dirs, files in os.walk(root_dir2):  # ./clip/train/
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(root_dir2 + dir):  # ./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".wav"):
                            self.data_list.append(os.path.join(root2, file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)  # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        f = wave.open(self.data_list[index], 'rb')  # 获取索引为index的图片的路径名
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = torch.tensor(waveData).unsqueeze(0)
        waveData = waveData.type(torch.FloatTensor)
        label = int(self.label_list[index])
        sample = (waveData, label)  # 根据图片和标签创建元组
        # if self.transform:
        #     sample = self.transform(sample)        # 对样本进行变换
        return sample
class Depression_wav_order_test14(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for root, dirs, files in os.walk(root_dir):                    #./clip/train/
            for dir in sorted(dirs):
                for root2, dirs2, files2 in os.walk(root_dir+dir):     #./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".wav"):
                            self.data_list.append(os.path.join(root2,file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)                       # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        f = wave.open(self.data_list[index], 'rb')  # 获取索引为index的图片的路径名
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = torch.tensor(waveData).unsqueeze(0)
        waveData = waveData.type(torch.FloatTensor)
        label = int(self.label_list[index])
        name = self.data_list[index].split('/')[-2]
        sample = (waveData, label,str(name))  # 根据图片和标签创建元组
        # if self.transform:
        #     sample = self.transform(sample)        # 对样本进行变换
        return sample
class Depression_wav_random_train(Dataset):
    def __init__(self, root_dir, transform=None,flag=3):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.data_list = []
        self.label_list = []
        for j in range(160):
            list_shuffle = []
            labels = os.listdir(root_dir)
            for i in range(len(labels)):
                list = []
                for root, dirs, files in os.walk(os.path.join(root_dir, labels[i])):
                    for file in files:
                        if file.endswith(".wav"):
                            list.append(os.path.join(root, file))
                random.shuffle(list)
                list = list[:10]
                list_shuffle += list
            random.shuffle(list_shuffle)
            self.data_list += list_shuffle
        for k in range(len(self.data_list)):
            self.label_list.append((self.data_list[k].split('/'))[flag])

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        f = wave.open(self.data_list[index], 'rb')  # 获取索引为index的图片的路径名
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = torch.tensor(waveData).unsqueeze(0)
        waveData = waveData.type(torch.FloatTensor)
        label = int(self.label_list[index])
        sample = (waveData, label)  # 根据图片和标签创建元组
        # if self.transform:
        #     sample = self.transform(sample)        # 对样本进行变换
        return sample
class Depression_wav_order_test(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for root, dirs, files in os.walk(root_dir):                    #./clip/train/
            for dir in sorted(dirs):
                for root2, dirs2, files2 in os.walk(os.path.join(root_dir, dir)):     #./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".wav"):
                            self.data_list.append(os.path.join(root2,file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)                       # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        f = wave.open(self.data_list[index], 'rb')  # 获取索引为index的图片的路径名
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = torch.tensor(waveData).unsqueeze(0)
        waveData = waveData.type(torch.FloatTensor)
        label = int(self.label_list[index])
        name = self.data_list[index].split('/')[-2]
        sample = (waveData, label,str(name))  # 根据图片和标签创建元组
        # if self.transform:
        #     sample = self.transform(sample)        # 对样本进行变换
        return sample
class Depression_wav_order_dev(Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir                   # 文件目录
        self.transform = transform                 # 变换
        #self.images = os.listdir(self.root_dir)    # 目录里的所有文件
        self.data_list=[]
        self.label_list=[]
        for root, dirs, files in os.walk(root_dir):                    #./clip/train/
            for dir in dirs:
                for root2, dirs2, files2 in os.walk(os.path.join(root_dir, dir)):     #./clip/train/00/
                    for file2 in files2:
                        if file2.endswith(".wav"):
                            self.data_list.append(os.path.join(root2,file2))  # ./clip/train/00/223_1/00235.jpg
                            self.label_list.append(dir)                       # 00

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        f = wave.open(self.data_list[index], 'rb')  # 获取索引为index的图片的路径名
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = torch.tensor(waveData).unsqueeze(0)
        waveData = waveData.type(torch.FloatTensor)
        label = int(self.label_list[index])
        sample = (waveData, label)  # 根据图片和标签创建元组
        # if self.transform:
        #     sample = self.transform(sample)        # 对样本进行变换
        return sample

def train_data_loader(root,batch_size,shuffle,flag):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    traindata = Depression_wav_random_train(root, transform=transform,flag=flag)  # 初始化类，设置数据集所在路径以及变换
    # print('batch_size: ', batch_size)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle,num_workers=4)  # 使用DataLoader加载数据
    return trainloader
def test_data_loader(root='./audio_wav_3s/test/',batch_size=1,shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testdata = Depression_wav_order_test(root, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=shuffle, num_workers=4,drop_last=True)  # 使用DataLoader加载数据
    return testloader
def val_data_loader(root='./audio_wav_3s/dev/',batch_size=64,shuffle=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testdata = Depression_wav_order_dev(root, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=shuffle, num_workers=4,drop_last=True)  # 使用DataLoader加载数据
    return testloader
def train_data_loader14(root,root2,batch_size,shuffle):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    traindata = Depression_wav_order_train14(root,root2, transform=transform)  # 初始化类，设置数据集所在路径以及变换
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle,num_workers=4)  # 使用DataLoader加载数据
    return trainloader

