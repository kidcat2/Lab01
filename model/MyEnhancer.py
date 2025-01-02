import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .AdaIN import AffineTransform, Style_Mod, Mapping, colorTransform3

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_2(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True))


def conv1x1_2(in_planes, out_planes, kernel_size=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True)) 

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

class SA_Block(nn.Module):
    def __init__(self, dim=256, dim_head=64, num_fmaps=4): 
        # x : dim, H, W
        # fmaps : num_fmaps, dim, h, w 
        # out : dim, H, W
        super(SA_Block, self).__init__()

        self.num_fmaps = num_fmaps
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * self.num_fmaps, bias=False)
        self.to_k = nn.Conv2d(in_channels=dim * self.num_fmaps, out_channels=dim_head * self.num_fmaps, kernel_size=1, groups=self.num_fmaps, bias=False)
        self.to_v = nn.Conv2d(in_channels=dim * self.num_fmaps, out_channels=dim_head * self.num_fmaps, kernel_size=1, groups=self.num_fmaps, bias=False)
        self.rescale = nn.Parameter(torch.ones(self.num_fmaps, 1, 1))

        self.ffn = nn.Linear(dim_head * self.num_fmaps, dim, bias=True)
    def forward(self, x, fmaps):
        """
        x : b c h w (4, 256, 32, 32)
        fmaps : b n c h w (4, 4, 256, 32, 32)
        # N : Style image 개수
        """
        B, N, C, H, W = fmaps.size() # [4, 4, 256, 32, 32]
        fmaps = fmaps.view(B, N*C, H, W) # (4, 4*256, 32, 32) = 4, 1024, 32, 32
        x_in = x.permute(0,2,3,1).reshape(B, H*W, C) # (4, 32*32, 256) = 4, 1024, 256

        q_in = self.to_q(x_in) # (4, 32*32, 256) = [4, 1024, 256]
        k_in = self.to_k(fmaps).permute(0,2,3,1).reshape(B, H*W, -1) # (4, 64*4, 32, 32) -> (4, 32, 32, 64*4) -> (4, 32*32, 64*4) = [4, 1024, 256]
        v_in = self.to_v(fmaps).permute(0,2,3,1).reshape(B, H*W, -1) 

        d = C // self.num_fmaps
        q = q_in.view(B, H*W, self.num_fmaps, d).permute(0, 2, 1, 3)  # (4, 4, 32*32, 64) = [4, 4, 1024, 64]
        k = k_in.view(B, H*W, self.num_fmaps, d).permute(0, 2, 1, 3)  
        v = v_in.view(B, H*W, self.num_fmaps, d).permute(0, 2, 1, 3)  

        q = q.transpose(-2, -1) # [4, 4, 64, 1024]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) # [4, 4, 64, 64]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1) 

        out = attn @ v # (4, 4, 32*2, 32*32) = [4, 4, 64, 1024] = (B, HEAD, C, H*W)
        out = out.permute(0, 3, 1, 2) # B, H*W, HEAD, C = 4, 1024, 4, 64
        out = out.reshape(B, H*W, self.num_fmaps*self.dim_head) # 4, 1024, 4*64
        out = out + q_in # 4, 1024, 256
        out = self.ffn(out) # 4, 1024, 256
        out = out + q_in # 4, 1024, 256
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)

        return out

class MyEnhancer(nn.Module):
    def __init__(self):
        super(MyEnhancer, self).__init__()

        self.control_point = 64
        self.feature_num = 256

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

        self.style_gap = nn.AdaptiveAvgPool2d((8,8))
        self.color_transform = colorTransform3(feature_num=self.feature_num)
        self.upconv = conv3x3(in_planes=256, out_planes=3)
        self.relu1 = nn.ReLU()

    def forward(self, x, latent): #

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
        #out = self.res6(out) # b x 256 x 16 x 16

        # out의 -1 ~ 1 사이의 값을 --> 0~1 사이로
        out = (out+1) / 2

        # curve mapping
        # latent = 4 x 256 x 32 x 32
        # color_map = B, C, Control_points = 4 x 256 x 16
        # params = B, C, A, B  , A*B = Control_points = 4 x 256 x 4 x 4
        B, _, _, _ = out.size()
        color_map_control = torch.linspace(0, 1, self.control_point)
        color_map_control = color_map_control.unsqueeze(0).unsqueeze(0)
        color_map_control = color_map_control.expand(B, self.feature_num, self.control_point) # 4 x 256 x 16
        color_map_control = color_map_control.cuda()
        
        params = self.style_gap(latent) # 4 x 256 x 4 x 4 

        out = self.color_transform(out, params, color_map_control) # 4 x 256 x 16 x 16
        out = F.interpolate(out, size=(256,256), mode='bicubic', align_corners=True)
        out = self.upconv(out) # 4 x 3 x 16 x 16
        out = self.relu1(out)
        
        
            
        return x + out 
    
class TEnhancer(nn.Module):
    def __init__(self):
        super(TEnhancer, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=1, bias=True)
        self.depth_conv =  nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True, groups=4)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1, bias=True)

        ## 
        self.conv_d01 = conv3x3_2(in_planes=3, out_planes=64, stride=2) # 64 128 128
        self.conv_d02 = conv3x3_2(in_planes=64, out_planes=128, stride=2) # 128 64 64

        ## Down : 128 64 64 --> 64 128 128
        self.sa_d1 = SA_Block(dim=128, dim_head=32)
        self.conv_d11 = conv3x3_2(in_planes=128, out_planes=256, stride=2) # 256 32 32
        self.conv_d12 = conv3x3_2(in_planes=256, out_planes=256, stride=1) # 256 32 32

        # d2
        self.sa_d2 = SA_Block(dim=256, dim_head=64)
        self.conv_d21 = conv3x3_2(in_planes=256, out_planes=512, stride=2) # 512 16 16
        self.conv_d22 = conv3x3_2(in_planes=512, out_planes=512, stride=1) # 512 16 16

        # d3
        self.sa_d3 = SA_Block(dim=512, dim_head=128)
        
        # to Dec
        self.conv_E = conv1x1_2(in_planes=512, out_planes=512, stride=1) # 512 16 16
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 256 32 32
        self.conv_D = conv1x1_2(in_planes=256, out_planes=256, stride=1) # 256 32 32

        ## Up
        self.conv_u1 = conv1x1_2(512, 256) # 256 32 32
        self.sa_u1 = SA_Block(dim=256, dim_head=64)

        # u2
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 128 64 64
        self.conv_u2 = conv1x1_2(256, 128) # 128 64 64
        self.sa_u2 = SA_Block(dim=128, dim_head=32) # 128 64 64 

        # Final
        self.conv3 = conv1x1_2(in_planes=128, out_planes=128, stride=1) 
        self.conv4 = conv1x1_2(in_planes=128, out_planes=3, stride=1)
        
        self.upconv = conv3x3(in_planes=256, out_planes=3)
        self.relu1 = nn.ReLU()

    def forward(self, x, latent): #

        B, C, H, W = x.size()
        latent1 = latent[0].view(B, -1, 128, 64, 64)        
        latent2 = latent[1].view(B, -1, 128, 64, 64)  
        latent3 = latent[2].view(B, -1, 256, 32, 32)        
        latent4 = latent[3].view(B, -1, 256, 32, 32)        
        latent5 = latent[4].view(B, -1, 512, 16, 16)        


        mean_c = x.mean(dim=1).unsqueeze(1) 
        illu_map = torch.cat([x, mean_c], dim=1) # B, 4, 256, 256
        illu_map = self.conv1(illu_map)
        illu_map = self.depth_conv(illu_map)
        illu_map = self.conv2(illu_map)
        illu_x = x*illu_map + x # 3 256 256
 
        ## 
        out = self.conv_d01(illu_x) # 64 128 128
        out = self.conv_d02(out) # 128 64 64

        ## Down
        f1 = self.sa_d1(out, latent1) # 128 64 64 
        f2 = self.conv_d11(f1) # 256 32 32
        f2 = self.conv_d12(f2) # 256 32 32

        f3 = self.sa_d2(f2, latent3) # 256 32 32
        f4 = self.conv_d21(f3) # 512 16 16
        f4 = self.conv_d22(f4) # 512 16 16

        in_dec = self.sa_d3(f4, latent5) # 512 16 16
        in_dec = self.conv_E(in_dec)  # 512 16 16
        in_dec = self.deconv1(in_dec) # 256 32 32
        in_dec = self.conv_D(in_dec)  # 256 32 32

        ## Up
        f5 = self.conv_u1(torch.cat((in_dec, f3), dim=1)) # 256 32 32
        f5 = self.sa_u1(f5, latent4) # 256 32 32
        
        f6 = self.deconv2(f5) # 128 64 64
        f6 = self.conv_u2(torch.cat((f6, f1), dim=1))# 128 64 64
        f6 = self.sa_u2(f6, latent2) # 128 64 64 

        dec_out = self.conv3(f6) # 128 64 64 
        dec_out = F.interpolate(out, size=(256,256), mode='bicubic', align_corners=True) # 128 256 256
        dec_out = self.conv4(dec_out) # 3 256 256 
         
        return illu_x + dec_out 