import numpy as np
import torch
import cv2
import logging, os
import math

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		if math.isnan(val) or math.isinf(val):
			print(f"Invalid value detected: {val}")  # 오류 로그 출력
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count





def tensor2img(img):
	img = img.squeeze(0)
	img = img.numpy().transpose((1, 2, 0))
	#img = ((img+1) * 127.5).astype(np.uint8)
	img = (img * 255).astype(np.uint8)
	img = np.array(img)

	return img

def dict2img(x): 
	"""
	Dictionary x
	- {'raw' : [raw_img1, raw_img2...], 'enhance' : [enhance_img1,...], 
		'gt' : [..] , 'pref' : [..], 'style' : [pref_style_name1, ....]}
	"""
	for key in x.keys():
		save_dir = f'imgsave/{key}'

		if key == 'style' : 
			save_dir = 'imgsave/style/style.txt'

			
			with open(save_dir, 'w') as f:
				l = x['style']
				for i, sn in enumerate(l) :
					f.write(f'{i}_{sn}\n')
							
			continue

		for i, img in enumerate(x[key]):
			path = os.path.join(save_dir, f'{i}.png')
			rimg = tensor2img(img)
			cv2.imwrite(path, rimg)


