import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from .PieNetEmbedNet import EmbedNet

def gram_matrix(img):
    (b, c, h, w) = img.size()
    img = img.view(b, c, h * w)
    gram = torch.bmm(img, img.transpose(1, 2))
    
    return gram / (c * h * w)

def gram_loss(y_hat, style):
    
    # style img : batch, img-count, c, h, w
    if style.dim() > 4 : 
        B, N, C, H, W = style.size()
        y_hat = y_hat.unsqueeze(1).expand(-1, N, -1, -1, -1)

        y_hat = y_hat.reshape(B * N, C, H, W)
        style = style.reshape(B * N, C, H, W)

    y_hat = gram_matrix(y_hat)
    style = gram_matrix(style)
    
    loss = F.mse_loss(y_hat, style)
    
    return loss

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img) 

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data   = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad   = False


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(Vgg19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        for x in range(5):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std  = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, X):
        h         = self.sub_mean(X)
        h_relu5_1 = self.slice1(h)
        return h_relu5_1

  
class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = Vgg19(requires_grad=False).to('cuda')

    def forward(self, out_image, gt_image, pref_image):
     
        out_feat = self.vgg19(out_image)
        gt_feat = self.vgg19(gt_image)

        lossRec = F.l1_loss(out_image, gt_image)
        lossPer = F.l1_loss(out_feat, gt_feat) * 0.4
        lossgram = gram_loss(out_image, pref_image) * 0.01
        lossTV = total_variation_loss(out_image) * 0.01

        loss = lossRec + lossPer + lossgram +lossTV 

        return loss


 