import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import sys
from tqdm    import tqdm
import torchvision.models as models
from PIL import Image

# StarEnhancer Mapping Network 
class Mapping(nn.Module):
    def __init__(self, latent_dim, channel):
        super(Mapping, self).__init__()
        
        self.channel = channel
        self.mlp_in = nn.Sequential(
			nn.Linear(latent_dim, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU()
		)
        
        self.mlp_out = nn.ModuleList()
        self.mlp_out.append(
            nn.Sequential(
                nn.Linear(512, 512),
                nn.PReLU(),
                #nn.Linear(512, 96),
                nn.Linear(512, self.channel * 32),#32),
                nn.Sigmoid()
            )
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, latent): 
        B, _, _, _, = x.shape # transform
        style = self.mlp_in(latent)
        style = self.mlp_out[0](style) # 4 x 512
        style = style.view(-1, 2, self.channel, 1, 1)
        
        scale = style[:, 0] # + 1 # [1, 256, 1, 1]
        shift = style[:, 1] # [1, 256, 1, 1]
        
        return x * scale + shift

# StyleGAN Model Mapping Network
class AffineTransform(nn.Module):
    def __init__(self, latent_dim, channel):
        super(AffineTransform, self).__init__()
        self.fc = nn.Linear(latent_dim, channel * 2)  
        self.channel = channel

    def forward(self, x, latent): # x : [1, 256, 1, 1], latent : [1, 512]
        # Learned Affine Transformation
        style = self.fc(latent) # [1, 512]
        style = style.view(-1, 2, self.channel, 1, 1) # [1, 2, 256, 1, 1]
        
        # AdaIN
        scale = style[:, 0] + 1 # [1, 256, 1, 1]
        shift = style[:, 1] # [1, 256, 1, 1]
        return x * scale + shift

class Style_Mod(nn.Module):
    def __init__(self, in_channel_list = [32, 64, 128, 256, 512] , latent_dim = 512):
        super(Style_Mod, self).__init__()
        
        self.affine_layers = nn.ModuleDict()
        self.channel_list = in_channel_list

        for channel in self.channel_list:
            self.affine_layers[str(channel)] = AffineTransform(latent_dim, channel)


    def forward(self, x, latent): # x : output  , latent : style-vector
        return self.affine_layers[str(x.shape[1])](x, latent)
    
