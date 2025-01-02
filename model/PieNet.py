import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

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
    
class PieNet(nn.Module):
    def __init__(self):
        super(PieNet, self).__init__()
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

        self.convD1_1 = conv1x1(512, 256)
        self.convD1_2 = conv1x1(512, 256)
        self.convD11 = conv3x3(256*4, 256)
        self.resD1 = ResBlock(in_channels=256, out_channels=256)

        self.convD2_1 = conv1x1(256, 128)
        self.convD2_2 = conv1x1(512, 128)
        self.convD22 = conv3x3(128*4, 128)
        self.resD2 = ResBlock(in_channels=128, out_channels=128)

        self.convD3_1 = conv1x1(128, 64)
        self.convD3_2 = conv1x1(512, 64)
        self.convD33 = conv3x3(64*4, 64)
        self.resD3 = ResBlock(in_channels=64, out_channels=64)

        self.convD4_1 = conv1x1(64, 64)
        self.convD4_2 = conv1x1(512, 64)
        self.convD44 = conv3x3(64*3, 64)
        self.resD4 = ResBlock(in_channels=64, out_channels=64)        

        self.convD = conv3x3(64, 3) 

    def forward(self, imgIn, pref):
        # enc
        feat0 = self.conv1(imgIn) # 256 256 64

        feat1 = self.res1(feat0)
        feat2 = self.res2(feat1) 
        feat2 = self.conv2(feat2) # 128 128 64

        feat3 = self.res3(feat2)
        feat4 = self.res4(feat3)
        feat4 = self.conv4(feat4) # 64 64 128

        feat5 = self.res5(feat4)
        feat6 = self.res6(feat5)
        feat6 = self.conv6(feat6) # 32 32 256

        feat7 = self.res7(feat6)
        feat8 = self.res8(feat7)
        feat8 = self.conv8(feat8) # 16 16 512

        outEnc = F.avg_pool2d(feat8, 16) # 1 1 512

        featPre = pref.unsqueeze(-1).unsqueeze(-1) # 1 1 512
        # featPre = featPre.repeat(imgIn.shape[0], 1, 1, 1)

        # dec

        """
        self.convD1_1 = conv1x1(512, 256)
		self.convD1_2 = conv1x1(512, 256)
		self.convD11 = conv3x3(256*4, 256)
		self.resD1 = ResBlock(in_channels=256, out_channels=256)

        
        """
        featD0 = outEnc
        featD1_1 = F.interpolate(self.convD1_1(featD0), scale_factor=32, mode='bicubic')
        featD1_2 = F.interpolate(self.convD1_2(featPre), scale_factor=32, mode='bicubic')

        featD1 = self.resD1(self.convD11(torch.cat((feat6, featD1_1, featD1_2), dim=1))) # 32 32 256
        featD2_1 = F.interpolate(self.convD2_1(featD1), scale_factor=2, mode='bicubic')
        featD2_2 = F.interpolate(self.convD2_2(featPre), scale_factor=64, mode='bicubic')

        featD2 = self.resD2(self.convD22(torch.cat((feat4, featD2_1, featD2_2), dim=1))) # 64 64 128
        featD3_1 = F.interpolate(self.convD3_1(featD2), scale_factor=2, mode='bicubic')
        featD3_2 = F.interpolate(self.convD3_2(featPre), scale_factor=128, mode='bicubic')        

        featD3 = self.resD3(self.convD33(torch.cat((feat2, featD3_1, featD3_2), dim=1))) # 128 128 64
        featD4_1 = F.interpolate(self.convD4_1(featD3), scale_factor=2, mode='bicubic')
        featD4_2 = F.interpolate(self.convD4_2(featPre), scale_factor=256, mode='bicubic')                

        featD4 = self.resD3(self.convD44(torch.cat((feat0, featD4_1, featD4_2), dim=1)))
        featD5_1 = F.interpolate(featD4, scale_factor=1, mode='bicubic')

        outDec = self.convD(featD5_1)

        imgOut = imgIn + outDec

        return imgOut, outDec