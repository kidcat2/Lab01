import os
import random
import numpy as np
import cv2
from glob import glob
from PIL import Image
import torch
import sys
from torch.utils.data import Dataset

# PieNet Style-Network Dataset
class StyleTrain(Dataset):
    def __init__(self, dir, path):
        self.dir = dir
        self.path = path
        self.sample = {}
        self.img_num = 0
        self.train_index = list(range(4500))
        self.test_index = list(range(500))
        

        for style_path in sorted(glob(os.path.join(dir, self.path ,'*'))):
            style_name = os.path.basename(style_path)
            self.sample[style_name] = sorted(os.listdir(os.path.join(style_path)))
            self.img_num = len(self.sample[style_name])

        #for key, value in self.sample.items() :
            #print(key, value)
    def __len__(self):
        return self.img_num
    
    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        data = {}
        
        for style_name, img_list in self.sample.items() :
            img = cv2.imread(self.dir  + self.path + '/' + style_name + '/' + img_list[idx])
            
            if img is None:
                sys.stdout.write(f"style_name : {style_name} / {idx}") 
            img_256 = np.array(Image.fromarray(img).resize((256, 256), Image.BICUBIC)).astype(np.float32) /127.5 -1.
            img_tensor = torch.from_numpy(img_256.transpose((2, 0, 1))).float()
            data[style_name] = img_tensor
        
        return data

# Our Model
class style_train(Dataset):
    def __init__(self, dir, path):
        self.dir = dir 
        self.path = path 
        self.sample = {} 
        self.img_num = 0 
        self.style_name = [] 
        self.train_index = list(range(4500))
        self.test_index = list(range(500))
        
        for style_path in sorted(glob(os.path.join(dir, self.path ,'*'))):
            style_name = os.path.basename(style_path)
            self.sample[style_name] = sorted(os.listdir(os.path.join(style_path)))
            self.img_num = len(self.sample[style_name])
            if style_name != 'raw':
                self.style_name.append(style_name)
            
    def __len__(self):
        return self.img_num
    
    def __getitem__(self, idx):
        """
        style : "Random style K"
        raw : "raw img x"
        gt : "K-style img x"
        pref : "Randomly select 10 images from K-style images (except img x)" 
        """

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        style = random.choice(self.style_name)

        raw = cv2.imread(self.dir  + self.path + '/raw/' + self.sample['raw'][idx])
        raw_256 = np.array(Image.fromarray(raw).resize((256, 256), Image.BICUBIC)).astype(np.float32) / 255.
        raw_tensor = torch.from_numpy(raw_256.transpose((2, 0, 1))).float()

        gt = cv2.imread(self.dir  + self.path + '/' + style + '/' + self.sample[style][idx])
        gt_256 = np.array(Image.fromarray(gt).resize((256, 256), Image.BICUBIC)).astype(np.float32) /  255.
        gt_tensor = torch.from_numpy(gt_256.transpose((2, 0, 1))).float()

        pref = []
        #"""
        if self.path == 'train':
            l = random.sample([i for i in self.train_index if i != idx], 4)
        else :
            l = random.sample([i for i in self.test_index if i != idx], 4)
        #"""

        
        for ridx in l :
            pref_img =  cv2.imread(self.dir  + self.path + '/' + style + '/' + self.sample[style][ridx])
            pref_img_256 = np.array(Image.fromarray(pref_img).resize((256, 256), Image.BICUBIC)).astype(np.float32) / 255.
            pref_tensor = torch.from_numpy(pref_img_256.transpose((2, 0, 1))).float()
            pref.append(pref_tensor)

        pref = torch.stack(pref)

        """ 1 style_img
        while True :
            if self.path == 'train':
                ridx = random.randint(0,4499)
            elif self.path == 'test' :
                ridx = random.randint(0,499)
            if ridx != idx : break

        pref_img =  cv2.imread(self.dir  + self.path + '/' + style + '/' + self.sample[style][ridx])
        pref_img_256 = np.array(Image.fromarray(pref_img).resize((256, 256), Image.BICUBIC)).astype(np.float32) /  255.
        pref = torch.from_numpy(pref_img_256.transpose((2, 0, 1))).float()
        #"""

        return raw_tensor, gt_tensor, pref, style