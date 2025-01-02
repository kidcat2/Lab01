import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .AdaIN import AffineTransform, Style_Mod, Mapping

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True))

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True))  

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels,  out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        fIn  = x
        fOut = self.conv1(x)
        fOut = self.conv2(fOut)
        fOut = fOut * self.res_scale + fIn
        return fOut

class AdaINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1, in_scale = 3):
        super(AdaINBlock, self).__init__()

        self.res_scale = res_scale
        self.res_channel = out_channels
        self.adain = Mapping(latent_dim=512, channel=out_channels)
        self.res = ResBlock(self.res_channel, self.res_channel, res_scale=self.res_scale)

        self.conv1 = conv1x1(in_channels,  out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.conv4 = conv3x3(out_channels * in_scale, out_channels)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        
    def forward(self, x, latent1, latent2, outEnc, up_scale = 2):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.adain(x, latent1)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.adain(x, latent2)
        x = self.conv3(x)
        x = self.act3(x)
        
        x = F.interpolate(x, scale_factor=up_scale, mode='bicubic') # delete conv
        x = torch.cat((x,outEnc), dim=1)
        x = self.conv4(x)
        x = self.act4(x)
        out = self.res(x)

        return out
         

class AdaINPieNet(nn.Module):
    def __init__(self):
        super(AdaINPieNet, self).__init__()
       
        # enc
        self.conv1 = conv3x3(3, 64)

        self.res1 = ResBlock(in_channels=64, out_channels=64, res_scale=1)
        self.res2 = ResBlock(in_channels=64, out_channels=64, res_scale=1)
        self.conv2 = conv3x3(64, 128, stride=2)

        self.res3 = ResBlock(in_channels=128, out_channels=128, res_scale=1)
        self.res4 = ResBlock(in_channels=128, out_channels=128, res_scale=1)
        self.conv4 = conv3x3(128, 256, stride=2)

        self.res5 = ResBlock(in_channels=256, out_channels=256, res_scale=1)
        self.res6 = ResBlock(in_channels=256, out_channels=256, res_scale=1)
        self.conv6 = conv3x3(256, 512, stride=2)

        self.res7 = ResBlock(in_channels=512, out_channels=512, res_scale=1)
        self.res8 = ResBlock(in_channels=512, out_channels=512, res_scale=1)
        self.conv8 = conv3x3(512, 512, stride=2)

        #dec
        self.adain1 = AdaINBlock(in_channels=512, out_channels=256, res_scale=1)
        self.adain2 = AdaINBlock(in_channels=256, out_channels=128, res_scale=1)
        self.adain3 = AdaINBlock(in_channels=128, out_channels=64, res_scale=1)
        self.adain4 = AdaINBlock(in_channels=64, out_channels=64, res_scale=1, in_scale = 2)     

        self.convD = conv3x3(64, 3) 

    def forward(self, img, latent): 
        """
        img : b x 3 x 256 x 256
        latent : b x 8 x 512
        """
            
        # enc
        feat0 = self.conv1(img) # b x 64 x 256 x 256

        feat1 = self.res1(feat0) # b x 64 x 256 x 256
        feat2 = self.res2(feat1) # b x 64 x 256 x 256 
        feat2 = self.conv2(feat2) # b x 128 x 128 x 128 

        feat3 = self.res3(feat2) # b x 128 x 128 x 128 
        feat4 = self.res4(feat3) # b x 128 x 128 x 128 
        feat4 = self.conv4(feat4) # b x 256 x 64 x 64 

        feat5 = self.res5(feat4) # b x 256 x 64 x 64 
        feat6 = self.res6(feat5) # b x 256 x 64 x 64 
        feat6 = self.conv6(feat6) # b x 512 x 32 x 32 

        feat7 = self.res7(feat6) # b x 512 x 32 x 32 
        feat8 = self.res8(feat7) # b x 512 x 32 x 32 
        feat8 = self.conv8(feat8) # b x 512 x 16 x 16 

        outEnc = F.avg_pool2d(feat8, 16) # b x 512 x 1 x 1 

        # dec AdaIN
        latent1 = latent[:, 0, :]
        latent2 = latent[:, 1, :]
        latent3 = latent[:, 2, :]
        latent4 = latent[:, 3, :]
        latent5 = latent[:, 4, :] 
        latent6 = latent[:, 5, :]
        latent7 = latent[:, 6, :]
        latent8 = latent[:, 7, :]

        featD0 = outEnc # b x 512 x 1 x 1 
        featD1 = self.adain1(featD0, latent1, latent2, feat6, up_scale = 32) # [1, 256, 32, 32] self, x, latent1, latent2, outEnc, up_scale = 2
        featD2 = self.adain2(featD1, latent3, latent4, feat4, up_scale = 2) # [1, 128, 64, 64]
        featD3 = self.adain3(featD2, latent5, latent6, feat2, up_scale = 2) # [1, 64, 128, 128]
        featD4 = self.adain4(featD3, latent7, latent8, feat0, up_scale = 2) # [1, 64, 256, 256]
        featD5 = F.interpolate(featD4, scale_factor=1, mode='bicubic') # [1, 64, 256, 256]

        outDec = self.convD(featD5) # [1, 3, 256, 256]

        imgOut = img + outDec

        return imgOut, outDec    


