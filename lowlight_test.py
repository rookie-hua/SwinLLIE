import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from einops import rearrange



def pad_image(image):
	w,h = image.size
	a,b = divmod(w,224)
	c,d = divmod(h,224)

	target_w = (a+1)*224 if b != 0 else a*224
	target_h = (c+1)*224 if d != 0 else c*224
		
	new_image = Image.new('RGB',(target_w,target_h),(0,0,0))
	new_image.paste(image,(0,0))
	return new_image,int(target_h/224),int(target_w/224),h,w


def lowlight(image_path,weight):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	image = Image.open(image_path)
	data_lowlight,p1,p2,h,w = pad_image(image)
	# data_lowlight = data_lowlight.resize((256,256),Image.ANTIALIAS)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	# data_lowlight = torchvision.transforms.functional.center_crop(data_lowlight,(256,256)) #中心剪裁
	data_lowlight = rearrange(data_lowlight,'c (h p1) (w p2)->(p1 p2) c h w',p1 = p1,p2 = p2).cuda()
	# data_lowlight = data_lowlight.cuda().unsqueeze(0)
	
	DCE_net = model.SwinLLIE().cuda()

	DCE_net.eval()
	DCE_net.load_state_dict(torch.load(weight))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)
	enhanced_image = rearrange(enhanced_image,'(p1 p2) c h w->c (h p1) (w p2)',p1 = p1,p2 = p2)
	enhanced_image = enhanced_image[:,:h,:w]

	end_time = (time.time() - start)
	# print("{:<10.5f}s".format(end_time))
	
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(result_path.split('\\')[0]):
		os.makedirs(result_path.split('\\')[0])
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		
		filePath = './data/test_data/'
		file_list = os.listdir(filePath)
		weight = './snapshots/Epoch99.pth'
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				print(image)
				lowlight(image, weight)
        