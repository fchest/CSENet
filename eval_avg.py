# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import logging
import os
import pickle
batch_size=64
logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)


def test(net,testloader,device):
    net.eval()
    batch_loss=0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i,data in enumerate(testloader):
            images, labels = data
            images, labels=images.to(device),labels.to(device)
            outputs = net(images)
            outputs=outputs.squeeze(-1)
            outputs=outputs.to(device)
            loss = criterion(outputs, labels.float())
            batch_loss += loss.item()
            batch_num=i+1
        mse=(batch_loss/batch_num)**0.5
    logger.info("test mse is: %5f" % mse)
    return mse

def test_num(net, net2, testloader,device):
    net.eval()
    batch_loss=0
    j=1
    tem_name="316_1"
    tem_label=0
    total_mse=0
    total_mae=0
    batch_num=0
    sum_outputs = torch.tensor([0])
    sum_outputs = sum_outputs.to(device)
    with torch.no_grad():
        for i,data in enumerate(testloader):
            images, labels,name = data
            name=str(name).split("'")[1]
            # print(name)
            if (tem_name==name):
                images, labels = images.to(device), labels.to(device)
                weighted = net(images)
                if batch_num == 0:
                    sum_weighted = weighted*0
                sum_weighted = sum_weighted + weighted
                # outputs = outputs.squeeze(-1)
                # outputs = outputs.to(device)
                # sum_outputs=sum_outputs+outputs
                # batch_loss += 1
                batch_num = batch_num + 1
            else:
                if batch_num == 0:
                    # predict=sum_weighted
                    predict = net2(images, sum_weighted)
                else:
                    weighted_avg = sum_weighted/batch_num
                
                predict = net2(images, weighted_avg)
                # print(predict.size())
                # sum_outputs = sum_outputs + predict
                logger.info("%s test label is : %d ,predict is: %5f" %
                            (tem_name,tem_label,float(predict)))
                total_mse = total_mse + math.pow(float(predict)-tem_label,2)
                total_mae = total_mae + abs(float(predict) - tem_label)
                j += 1
                batch_loss  = 0
                batch_num   = 0
                sum_outputs = 0
                tem_name    = name
                tem_label   = labels
    # predict = sum_outputs / batch_num
    weighted_avg = sum_weighted/batch_num
    predict = net2(images, weighted_avg)
    logger.info("%s test label is : %d ,predict is: %5f" % (name, labels, predict))
    total_mse = total_mse + math.pow(float(predict)-int(labels),2)
    total_mae = total_mae + abs(float(predict) - tem_label)
    total_mse=math.sqrt(total_mse/j)
    total_mae = (total_mae / j).item()
    logger.info(total_mse)
    logger.info(total_mae)
    return total_mse,total_mae

def figure(net,testloader,device):
    net.eval()
    tem_name="317_4"
    j=0
    with torch.no_grad():
        for i,data in enumerate(testloader):
            images, labels, name = data
            if(tem_name!=name):
                j=0
                tem_name=name
            images=images.to(device)
            y,atty = net(images)
            root="../figure/y/"+str(int(labels))+"/"+str(name[0])
            if not os.path.exists(root):
                os.makedirs(root)
            y_output = open(root+"/"+str(j)+".pkl", 'wb')
            pickle.dump(y, y_output)
            y_output.close()

            attroot = "../figure/atty/" + str(int(labels)) + "/" + str(name[0])
            if not os.path.exists(attroot):
                os.makedirs(attroot)
            atty_output = open(attroot + "/" + str(j) + ".pkl", 'wb')
            pickle.dump(atty, atty_output)
            atty_output.close()
            j+=1
            