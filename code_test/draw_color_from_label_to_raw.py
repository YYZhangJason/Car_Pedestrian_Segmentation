#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: draw_color_from_label_to_raw.py
@time: 3/21/20 4:27 PM
@desc:
'''
import cv2
import numpy as np
from PIL import Image
import os
def chroma_blend(image, img_id):
	# 0:empty,1:road,2:ped,3:unused,4:car
	palette_ = [[0, 0, 0], [160, 32, 240], [255, 0, 0], [255, 0, 0], [0, 0, 255], [255, 165, 0]]
	palette = np.zeros((256, 3),np.uint)
	for i, p in enumerate(palette_):
		palette[i, 0] = p[0]
		palette[i, 1] = p[1]
		palette[i, 2] = p[2]
	palette = palette[..., ::-1]

	color = palette[img_id.ravel()].reshape(image.shape)

	image_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV)
	image_y,image_u,image_v = cv2.split(image_yuv)
	color_yuv = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_BGR2YUV)
	color_y,color_u,color_v = cv2.split(color_yuv)
	image_y = np.uint8(image_y)
	color_u = np.uint8(color_u)
	color_v = np.uint8(color_v)
	image_yuv = cv2.merge((image_y,color_u,color_v))
	image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
	return image
def get_img_label(raw_img_path,label_img_path):
	img_raw = cv2.imread(raw_img_path)
	img_label = Image.open(label_img_path)
	img_raw = np.array(img_raw).astype(np.uint8)
	img_label = np.array(img_label).astype(np.uint8)
	img_raw = cv2.resize(img_raw, (768, 384), interpolation=cv2.INTER_LINEAR)
	img_label = Image.fromarray(img_label, 'P')
	img_label = img_label.resize([768, 384], Image.NEAREST)
	img_label = np.array(img_label, dtype=np.uint8)
	return img_raw , img_label
#只要改这三个路径就行
# path_raw = "/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/raw/right"#这是原图路径
# path_label ='/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/color_draw/label'#这是标签路径
# save_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/color_draw/draw'#这是保存路径
# if not os.path.exists(save_dir):
# 	os.makedirs(save_dir)
# path_raw_list = os.listdir(path_raw)
# for name in path_raw_list:
# 	path_raw_single_name =os.path.join(path_raw,name)
# 	path_label_single_name = os.path.join(path_label,name)
# 	print(path_raw_single_name)
# 	print(path_label_single_name)
# 	cv2.imread(path_label_single_name)
# 	img_raw, img_label = get_img_label(path_raw_single_name, path_label_single_name)
# 	draw_img = chroma_blend(img_raw, img_label)
# 	save_path = os.path.join(save_dir,name)
# 	cv2.imwrite(save_path,draw_img)
#------------------上边是直接读取文件夹里面的图片，下面是通过txt文档的路径进行读取---------------
txt_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/only_test_img/bsd_day'
path_raw  = open(os.path.join(txt_dir,'train_img_R.txt'), 'r').readlines()
path_label = open(os.path.join(txt_dir,'train_label_R.txt'), 'r').readlines()
save_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/only_test_img/bsd_day/valiate/'
for i in range(len(path_raw)):
	if len(path_label)!=len(path_raw):
		print("raw and label have some problems!!")
		continue
	img_raw, img_label = get_img_label(path_raw[i].strip(), path_label[i].strip())
	draw_img = chroma_blend(img_raw, img_label)
	save_path = os.path.join(save_dir,str(i)+'.png')
	cv2.imwrite(save_path,draw_img)