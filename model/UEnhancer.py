import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .AdaIN import AffineTransform, Style_Mod, Mapping

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AdainResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1, in_scale = 3, downsample=None):
        super(AdainResBlk, self).__init__()

        self.res_scale = res_scale
        self.res_channel = out_channels
        self.adain = Mapping(latent_dim=512, channel=out_channels)
        
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(out_channels, out_channels)
        self.scale = nn.Parameter(torch.ones(1))

        self.downsample = downsample
        
    def forward(self, x, latent, up_scale = 2):
        identity = x

        out = self.actv1(x + self.bias1a)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out + self.bias1b)

        out = self.adain(out, latent)

        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)

        out = out * self.scale

        out += identity

        return out


class UEnhancer(nn.Module):
    def __init__(self):
        super(UEnhancer, self).__init__()

        self.cd = 256
        self.cl = 64
        self.dims = 64 * 9

        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.adain1 = AdainResBlk(in_channels=128, out_channels=128)
        self.down1 = conv3x3(128, 256, stride=2)

        self.adain2 = AdainResBlk(in_channels=256, out_channels=256)
        self.down2 = conv3x3(256, 512, stride=2)

        self.adain3 = AdainResBlk(in_channels=512, out_channels=512)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv1x1(512, 1024)
        self.actv2 = nn.PReLU()
        self.conv3 = conv1x1(1024, 512)
        self.actv3 = nn.PReLU()
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv4 = conv3x3(1024, 512)
        self.adain4 = AdainResBlk(in_channels=512, out_channels=512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5 = conv3x3(512, 256)
        self.adain5 = AdainResBlk(in_channels=256, out_channels=256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = conv3x3(256, 128)
        self.adain6 = AdainResBlk(in_channels=128, out_channels=128)

        self.convD = conv3x3(128, 3)


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512, self.dims)

    def curve(self, x, func, depth): # func = [1,9,256], depth = 256
        x_ind = x * (depth - 1) # x_ind = [1,9,256,256]
        x_ind = x_ind.long().flatten(2).detach() # x_ind = [1,9,65536]
        out = torch.gather(func, 2, x_ind) # device-side asser error!!
        return out.reshape(x.size())
    
    def forward(self, x, latent):

        latent1 = latent[:, 0, :]
        latent2 = latent[:, 1, :]
        latent3 = latent[:, 2, :]
        latent4 = latent[:, 3, :]
        latent5 = latent[:, 4, :]
        latent6 = latent[:, 5, :]

        out = x # b x 3 x 256 x 256
        out = self.conv1(out) # b x 128 x 128 x 128
        out = self.actv(out + self.bias1) # b x 128 x 128 x 128
        out = self.maxpool(out) # b x 128 x 64 x 64

        f1 = self.adain1(out, latent6) # b x 128 x 64 x 64
        f2 = self.down1(f1) # b x 256 x 32 x 32

        f2 = self.adain2(f2, latent4) # b x 256 x 32 x 32
        f3 = self.down2(f2) # b x 512 x 16 x 16

        f3 = self.adain3(f3, latent2) # b x 512 x 16 x 16

        outEnc =  self.maxpool2(f3) # b x 512 x 8 x 8 
        outEnc = self.conv2(outEnc) # b x 1024 x 8 x 8
        outEnc = self.actv2(outEnc) # b x 1024 x 8 x 8
        inDec = self.conv3(outEnc) # b x 512 x 8 x 8
        inDec = self.actv3(inDec) # b x 512 x 8 x 8
        
        f4 = self.deconv1(inDec) # b x 512 x 16 x 16
        f4 = self.conv4(torch.cat((f4, f3), dim=1)) # b x 512 x 16 x 16
        f4 = self.adain4(f4, latent1) # b x 512 x 16 x 16

        f5 = self.deconv2(f4) # b x 256 x 32 x 32
        f5 = self.conv5(torch.cat((f5, f2), dim=1)) # b x 256 x 32 x 32
        f5 = self.adain5(f5, latent3) # b x 256 x 32 x 32

        f6 = self.deconv3(f5) # b x 128 x 64 x 64
        f6 = self.conv6(torch.cat((f6, f1), dim=1)) # b x 128 x 64 x 64
        f6 = self.adain6(f6, latent5) # b x 128 x 64 x 64

        outDec = self.convD(f6) # b x 3 x 64 x 64
        outDec = F.interpolate(outDec, size=(256, 256), mode='bilinear', align_corners=True) # b x 3 x 256 x 256

        return x + outDec