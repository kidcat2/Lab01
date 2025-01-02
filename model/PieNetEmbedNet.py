import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True))


def conv1x1(in_planes, out_planes, kernel_size=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), bias=True), nn.LeakyReLU(0.1, inplace=True))                      


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*8*8, 512*1*1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])      

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out      


class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.build()

    def build(self):
        self.prefVec = nn.ParameterDict()
        self.prefVec["fA_"] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec["fB_"] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec["fC_"] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec["fD_"] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec["fE_"] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc1'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc2'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc3'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc4'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc5'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc6'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc7'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fc8'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp1'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp2'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp3'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp4'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp5'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp6'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp7'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp8'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp9'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp10'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp11'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp12'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp13'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp14'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))
        self.prefVec['fp15'] = nn.Parameter(F.normalize(torch.randn((1, 512*1*1)).cuda(), p=2, dim=1))

        self.embedNet = ResNet18()

    def forward(self, iP):
        B, N, _, _, _ = iP.size()
        iP = iP.view(-1, 3, 256, 256) # b*img개수, 3, 256, 256 [4*4, 3, 256, 256]
        fP = self.embedNet(iP)  # [16, 512, 1, 1]
        #fN = self.embedNet(iN)

        fP = F.normalize(fP, p=2, dim=1) # [16, 512, 1, 1]
        fP = fP.view(B,-1,512,1,1) # [4, 4, 512, 1, 1]
        fP = fP.squeeze(-1).squeeze(-1) # [4, 4, 512]
        fP = fP.mean(dim=1) # PieNet Enhancer 용, [4, 512]
        #fN = F.normalize(fN, p=2, dim=1)
        
        return fP#, fN[:, :, 0, 0]