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

# Color Transform
class colorTransform3(nn.Module):
    def __init__(self, feature_num, control_point=64, offset_param=0.04):#config=0):
        super(colorTransform3, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        #self.config = config
        self.feature_num = feature_num #config.feature_num
        self.epsilon = 1e-8
        self.offset_param = offset_param
        """
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param
        """
    def forward(self, org_img, params, color_map_control):
        N, C, H, W = org_img.shape
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
        color_map_control_y = color_map_control_x + params
        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2) # N x F x 17
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2) # N x F x 17
        img_reshaped = org_img.reshape(N, self.feature_num, -1) # rimg = N x F x ?
        img_reshaped_val = img_reshaped * (self.control_point-1) # v = i*15
        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1)) # v2 = floor(i * 15)
        img_reshaped_index = img_reshaped_index.type(torch.int64) # v2 = int(floor(i * 15))
        img_reshaped_index_plus = img_reshaped_index + 1 # v2p = v2 + 1
        img_reshaped_coeff = img_reshaped_val - img_reshaped_index #  z = v - v2
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff # z2 = 1 - z
        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index) # 4, 256, 65  <---  N, 256, 256
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)
        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped


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
    
