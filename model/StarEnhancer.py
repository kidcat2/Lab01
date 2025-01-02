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


class ResBlk(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlk, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.actv1(x + self.bias1a)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out + self.bias1b)

        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)

        out = out * self.scale

        out += identity

        return out

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
         

class StarEnhancer(nn.Module):
    def __init__(self):
        super(StarEnhancer, self).__init__()

        self.cd = 256
        self.cl = 64
        self.dims = 64 * 9

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1 = ResBlk(inplanes=64, planes=64, stride=1)
        self.res2 = ResBlk(inplanes=64, planes=64, stride=1)
        self.res3 = ResBlk(inplanes=64, planes=128, stride=1, downsample=conv1x1(64, 128, stride=1))
        self.res4 = ResBlk(inplanes=128, planes=128, stride=2, downsample=conv1x1(128, 128, stride=2))
        self.res5 = ResBlk(inplanes=128, planes=256, stride=1, downsample=conv1x1(128, 256, stride=1))
        self.res6 = ResBlk(inplanes=256, planes=256, stride=2, downsample=conv1x1(256, 256, stride=2))

        self.adain1 = AdainResBlk(in_channels=256, out_channels=256)
        self.adain2 = AdainResBlk(in_channels=256, out_channels=256)
        self.adain3 = AdainResBlk(in_channels=256, out_channels=512, stride=2, downsample=conv1x1(256, 512, stride=2))
        self.adain4 = AdainResBlk(in_channels=512, out_channels=512)
        #self.adain5 = AdainResBlk(in_channels=512, out_channels=512)
        ##self.adain6 = AdainResBlk(in_channels=512, out_channels=512)
        #self.adain7 = AdainResBlk(in_channels=512, out_channels=512)
        #self.adain8 = AdainResBlk(in_channels=512, out_channels=512)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512, self.dims)

    def curve(self, x, func, depth): # func = [1,9,256], depth = 256
        x_ind = x * (depth - 1) # x_ind = [1,9,256,256]
        x_ind = x_ind.long().flatten(2).detach() # x_ind = [1,9,65536]
        out = torch.gather(func, 2, x_ind) 
        return out.reshape(x.size())
    
    def forward(self, x, latent):

        B, _, H, W = x.size()

        out = x

        # CurveEncoder
        out = self.conv1(out) # b x 64 x 128 x 128
        out = self.actv(out + self.bias1)
        out = self.maxpool(out) # b x 64 x 64 x 64

        out = self.res1(out) # b x 64 x 64 x 64
        out = self.res2(out) # b x 64 x 64 x 64
        out = self.res3(out) # b x 128 x 64 x 64
        out = self.res4(out) # b x 128 x 32 x 32
        out = self.res5(out) # b x 256 x 32 x 32
        out = self.res6(out) # b x 256 x 16 x 16

        latent1 = latent[:, 0, :]
        latent2 = latent[:, 1, :]
        latent3 = latent[:, 2, :]
        latent4 = latent[:, 3, :]

        out = self.adain1(out, latent1)
        out = self.adain2(out, latent2)
        out = self.adain3(out, latent3)
        out = self.adain4(out, latent4)

        out = self.gap(out).flatten(1) # 4 x 512
        out = self.fc(out + self.bias2) # 4 x 576
        out = out.view(B, 9, self.cl, 1) # 4 x 9 x 64 x 1

        curves = F.interpolate(
            out, (self.cd, 1), 
            mode='bicubic', align_corners=True
        ).squeeze(3) # [4, 9, 256]

        residual = self.curve(
            x.repeat(1, 3, 1, 1), curves, self.cd
        ).view(B, 3, 3, H, W).sum(dim=2) # 4 x 3 x 256 x 256

        return x + residual
    


    