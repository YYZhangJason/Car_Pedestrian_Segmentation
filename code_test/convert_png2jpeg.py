#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: convert_png2jpeg.py
@time: 4/16/20 5:54 PM
@purpose:将png图片转换成jpeg格式进行训练，看结果是否会有所改变。
'''
import os
import cv2
img_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_test_txt/train_all'
img_r_txt = open(os.path.join(img_dir, 'train_img_R.txt'),'r').readlines()
img_l_txt = open(os.path.join(img_dir, 'train_img_L.txt'),'r').readlines()
train_all_txt = img_r_txt+img_l_txt
num_frm = len(train_all_txt)
img_path = train_all_txt[0].strip()
print(num_frm)
for frm_id in range(num_frm):
	if frm_id%250 ==0:
		print(frm_id)
	img_path = train_all_txt[frm_id].strip()
	img = cv2.imread(img_path)
	save_dir = img_path.strip('.png') + '.jpg'
	cv2.imwrite(save_dir,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
