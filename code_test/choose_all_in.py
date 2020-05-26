#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: choose_all_in.py
@time: 4/14/20 4:07 PM
@purpose:批量移动指定的文件
'''
import os

import shutil

#shutil.move('/media/soterea/Data_ssd/work/YYZhang/data/2020-03-16-14-57-06-BSD_61_left.png','/media/soterea/Data_ssd/work/YYZhang/2020-03-16-14-57-06-BSD_61_left.png')
raw_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/test_data/raw_img'
label_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/test_data/gt_color'
save_raw_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/only_test_img/bsd_day/raw_img'
save_label_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/only_test_img/bsd_day/gt_color'
count = 0
for side in os.listdir(raw_path):
	raw_path_side = os.path.join(raw_path,side)
	label_path_side = os.path.join(label_path,side)
	save_raw_path_side = os.path.join(save_raw_path,side)
	save_label_path_side = os.path.join(save_label_path,side)
	for name_img in os.listdir(raw_path_side):
		if name_img in os.listdir(label_path_side):
			raw_img_path = os.path.join(raw_path_side,name_img)
			label_img_path = os.path.join(label_path_side,name_img)
			save_raw_img_path = os.path.join(save_raw_path_side,name_img)
			save_label_img_path = os.path.join(save_label_path_side,name_img)
			# print(raw_img_path)
			# print(label_img_path)
			# print(save_raw_img_path)
			# print(save_label_img_path)
			shutil.move(raw_img_path,save_raw_img_path)
			shutil.move(label_img_path,save_label_img_path)


