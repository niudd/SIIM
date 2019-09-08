import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from .common import *

import numpy as np
import pandas as pd

import loss.lovasz_losses as L
from loss.losses import dice_loss, FocalLoss
from metrics import iou_pytorch, dice


class R2AttU_Net(nn.Module):
#     def __init__(self,img_ch=3,output_ch=1,t=2,debug=False):
#         super().__init__()
        
#         self.debug = debug
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

#         self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
    def __init__(self,img_ch=3,output_ch=1,t=2,debug=False):
        super().__init__()
        
        self.debug = debug
        
        #self.resnet = torchvision.models.resnet34(pretrained=True)

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

#         self.conv1 = nn.Sequential(
#             self.resnet.conv1,
#             self.resnet.bn1,
#             self.resnet.relu,
#             #self.resnet.maxpool,
#         )# 64
#         self.encoder2 = self.resnet.layer1
#         self.encoder3 = self.resnet.layer2
#         self.encoder4 = self.resnet.layer3
#         self.encoder5 = self.resnet.layer4
        
        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t, is_first_layer=False)#downsize 512 to 256

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=64, same=False)#the trick, set True, do not upsample, stay 256
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        #self.Up1 = up_conv(ch_in=64,ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)        
        if self.debug:
            print('input: ', x.size())
        
        # encoding path
        x1 = self.RRCNN1(x)#x1 = self.conv1(x)
        if self.debug:
            print('RRCNN1: ', x1.size())

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        #x2 = self.encoder2(x1)
        if self.debug:
            print('RRCNN2: ', x2.size())
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        #x3 = self.encoder3(x2)
        if self.debug:
            print('RRCNN3: ', x3.size())

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        #x4 = self.encoder4(x3)
        if self.debug:
            print('RRCNN4: ', x4.size())

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        #x5 = self.encoder5(x4)
        if self.debug:
            print('RRCNN5: ', x5.size())

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        if self.debug:
            print('====decoder====')
            print('Up_RRCNN5: ', d5.size())
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)
        if self.debug:
            print('Up_RRCNN4: ', d4.size())

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)
        if self.debug:
            print('Up_RRCNN3: ', d3.size())

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)
        if self.debug:
            print('Up_RRCNN2: ', d2.size())
        
#         d2 = self.Up1(d2)
#         if self.debug:
#             print('Up1: ', d2.size())

        d1 = self.Conv_1x1(d2)
        if self.debug:
            print('logit: ', d1.size())

        return d1
    
    ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""        
        Loss_FUNC = nn.BCEWithLogitsLoss()
        #Loss_FUNC = FocalLoss(alpha=1, gamma=2, logits=True, reduce=True)
        #bce_loss = Loss_FUNC(logit, truth)
        loss = Loss_FUNC(logit, truth)
        
        #loss = L.lovasz_hinge(logit, truth, ignore=None)#255
        #loss = L.symmetric_lovasz(logit, truth)
        #loss = 0.66 * dice_loss(logit, truth) + 0.33 * bce_loss
        #loss = dice_loss(logit, truth)
        return loss

    def metric(self, logit, truth):
        """Define metrics for evaluation especially for early stoppping."""
        #return iou_pytorch(logit, truth)
        return dice(logit, truth)

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError
