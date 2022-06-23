# -*- coding: utf-8 -*-
import torch
import logging
import os
from torchsummary import summary
#from resnet import *
from pathlib import Path
from eval import test,test_num
from dc_crn_test_attention import DCCRN
from train import train
from model import *
from dataload_vad import *
classes_num=1
logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

def test_attention_vis(net,testloader,device):
    net.eval()
    batch_loss=0
    j=1
    sum_outputs = torch.tensor([0])
    sum_outputs = sum_outputs.to(device)
    count = 0
    count1 = 0
    with torch.no_grad():
        for i,data in enumerate(testloader):
            images, labels,name = data
            if name[0] == '220_3':
                count1 += 1
                images, labels = images.to(device), labels.to(device)
                stats, weighted = net(images)
                weighted_path = os.path.join('attention_vis_220_3/', str(count1)+'_weighted.npy')
                # stats_path = os.path.join('attention_vis_316_1/', str(i)+'_stats.npy')
                # np.save(stats_path, stats.cpu())
                np.save(weighted_path, weighted.cpu())
                print(count1, name[0])
            # if name[0] == '237_1':
                # count += 1
                # images, labels = images.to(device), labels.to(device)
                # stats, weighted = net(images)
                # weighted_path = os.path.join('attention_vis_237_1/', str(count)+'_weighted.npy')
                # # stats_path = os.path.join('attention_vis_246_1/', str(i)+'_stats.npy')
                # # np.save(stats_path, stats.cpu())
                # np.save(weighted_path, weighted.cpu())
                # print(count, name[0])
    
    # return total_mse,total_mae

def load_net(net,model_pkl):
    logger.info("load:%s"%model_pkl)
    net.load_state_dict(torch.load(model_pkl))
    return net
def count_parameters(model):
    parameters_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(parameters_sum)

def test_3s(data_root, model_path):
    # test_root = os.path.join(data_root, 'test')
    test_data=test_data_loader(data_root,batch_size=1,shuffle=False)
    net = DCCRN(rnn_units=256,use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[0])      #GPU配置
    net.to(device)
    net = load_net(net, model_path)
    test_attention_vis(net=net, testloader=test_data, device=device)
    # for i in range(100,0,-1):
        # path="3s_1d_"+str(i)+".pkl"
        # my_file = Path("../pkl/"+path)
        # if my_file.is_file():
            # net=load_net(net,path)
            # test_num(net=net, testloader=test_data, device=device)


if __name__ == "__main__":
    data_root = '/data3/fancunhang/Depression/audio_good_without_move/AVEC2013_3s/test/'
    # model_path = 'exp/best_model.pkl'
    model_path = 'exp_0.002/checkpoint/best_model_epoch_75.pkl'
    # model_path = 'exp/checkpoint/model_epoch_96.pkl'
    test_3s(data_root, model_path)


