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

from dataset.dataloader import style_train
from models.loss import TotalLoss
from models.utils import *

from models.model import AdaINPieNet, PieNet, StarEnhancer, UEnhancer, MyEnhancer, TEnhancer
from models.model import pSpEncoder, EmbedNet

# args
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Adobe5K/', type=str, help='data directory')
parser.add_argument('--save_dir', default='save', type=str, help='save directory')
parser.add_argument('--encoder_lr', default=1e-4, type=float, help='Llarning rate')
parser.add_argument('--enhancer_lr', default=1e-4, type=float, help='Llarning rate')
parser.add_argument('--epochs', default=200, type=int, help='training epochs')
parser.add_argument('--num_workers', default=4, type=int, help='workers')
parser.add_argument('--train_batch_size', default=4, type=int, help='train batch size')
parser.add_argument('--valid_batch_size', default=1, type=int, help='valid batch size')
parser.add_argument('--eval_freq', default=5, type=int, help='valid batch size')
parser.add_argument('--num_pref', default=1, type=int, help='pref imgs')

args = parser.parse_args()

# train
def train(train_loader, encoder, enhancer, criterion, optimizer):
    losses = AverageMeter()

    encoder.train()
    enhancer.train()
    
    for (raw, gt, pref, style) in tqdm(train_loader):
        raw = raw.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        pref = pref.cuda(non_blocking=True) 
        
        latent = encoder(pref)        
        y_hat = enhancer(raw, latent) 

        
        loss = criterion(y_hat, gt, pref)
        
        #print(x.item())
        
        losses.update(loss.item(), args.train_batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

# valid
def valid(valid_loader, encoder, enhancer):
    PSNR = AverageMeter()
    
    torch.cuda.empty_cache()

    img_save = {'raw' : [] , 'gt' : [] , 'pref' : [] ,'enhance' : [] , 'style' : [] }
    encoder.eval()
    enhancer.eval()

    for (raw, gt, pref, style) in tqdm(valid_loader):
        raw = raw.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        pref = pref.cuda(non_blocking=True)

        with torch.no_grad():
            latent = encoder(pref)
            y_hat = enhancer(raw, latent) 
            y_hat = y_hat.clamp_(0, 1)

        img_save['raw'].append(raw.cpu())
        img_save['enhance'].append(y_hat.cpu())
        img_save['gt'].append(gt.cpu())
        img_save['style'].append(style[0])

        mse_loss = F.mse_loss(y_hat, gt, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), args.valid_batch_size)

    dict2img(img_save)

    return PSNR.avg


if __name__ == '__main__':
    # Record
    train_log = "Logs/Train_Log.txt"
    valid_log = "Logs/Valid_Log.txt"
    best_log = "Logs/Best_Log.txt"

    # Encoder (스타일 추출기)
    psp = pSpEncoder() 
    psp.cuda()

    # Enahncer (이미지 개선)
    enhancer = TEnhancer()
    enhancer.cuda()
    
    # loss
    criterion = TotalLoss()

    # dataloader
    train_dataset = style_train(args.data_dir,"train")
    train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle = True, num_workers=args.num_workers)
    
    valid_dataset = style_train(args.data_dir,"test")
    valid_loader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, shuffle = False, num_workers=1)

    # optimzer & scheduler
    optimizer = torch.optim.Adam([
      		{'params': psp.parameters(), 'lr': args.encoder_lr },
			{'params': enhancer.parameters(), 'lr': args.enhancer_lr}
		])	

    scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_enhancer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0

    # train
    for epoch in range(args.epochs + 1):
        loss = train(train_loader, psp, enhancer, criterion, optimizer)

        print('Train [{0}]\t'
                'Loss: {loss:.4f}\t '
                'Best Val PSNR: {psnr:.3f}'.format(epoch, loss=loss, psnr=best_psnr))
        
        with open(train_log,"a") as file:
                file.write('Train [{0}]\t'
                'Loss: {loss:.4f}\t '
                'Best Val PSNR: {psnr:.3f}\n'.format(epoch, loss=loss, psnr=best_psnr))

        scheduler_encoder.step()
        scheduler_enhancer.step()
        
        # Log & Save
        if epoch % args.eval_freq == 0: 
            avg_psnr = valid(valid_loader, psp, enhancer)
            print('Valid: [{0}]\tPSNR: {psnr:.3f}'.format(epoch, psnr=avg_psnr))

            with open(valid_log,"a") as file:
                file.write('Valid: [{0}]\tPSNR: {psnr:.3f}\n'.format(epoch, psnr=avg_psnr))

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr

                with open(best_log,"a") as file:
                    file.write('Valid: [{0}]\tPSNR: {psnr:.3f}\n'.format(epoch, psnr=best_psnr))

                torch.save({'state_dict': psp.state_dict()}, 
                            os.path.join(args.save_dir, 'psp.pth.tar'))
                torch.save({'state_dict': enhancer.state_dict()}, 
                            os.path.join(args.save_dir, 'enhancer.pth.tar'))
            
        
        