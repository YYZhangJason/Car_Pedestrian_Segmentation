#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: get_txt_to_write.py
@time: 3/24/20 6:18 PM
@purpose:按行读取txt中数据，并且将其写入指定文件中
'''
import os
from PIL import Image
#-----------------------批量读取----------------------------
txt_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_test_txt/orignal'
#
# file = open('/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/20200323/bsd_day/test_img_L.txt')
f_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_test_txt/test_all'
for txt_name in os.listdir(txt_path):
	file = open(os.path.join(txt_path,txt_name))
	print os.path.join(txt_path,txt_name)
	while 1:
		lines = file.readlines(100000)
		f = open(os.path.join(f_path,txt_name),'a')
		#open
		#r:只读，r+：读写，不创建
		#w：新建只写，w+新建读写，w,w+都会将文件内容清零
		#a:附加写方式打开，不可读 a+:附加写方式打开
		if not lines:
			break
		for line in lines:
			f.write(line)
	file.close()
#---------------------------分割线-------------------------------
#-----------------------读取目录并将晚上的路径去除-----------------------------
#
# txt_path = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/train_txt/all_v.T_LR_onlyday_768x384.01/test/test_img_R.txt'
# txt_open = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/train_txt/all_v.T_LR_onlyday_768x384.01/test_img_R.txt'
# file = open(txt_path)
# count = 0
# while 1:
# 	lines = file.readlines(100000)
# 	print(lines)
# 	# f = open(txt_open,'a')
# 	if not lines:
# 		break
# 	# for line in lines:
# 	# 	if 'bsd_day' in line:
# 	# 		count +=1
# 	# 		print(count)
# 	# 		print line
# 	# 		f.write(line)
# file.close()
# #--------------------挑选指定txt中的图片进行保存-------------------------
# txt_dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/train_txt/all_20191217'
# img_l_list = open(os.path.join(txt_dir,'test_img_L.txt'), 'r').readlines()
# label_l_list = open(os.path.join(txt_dir,'test_label_L.txt'), 'r').readlines()
# img_r_list = open(os.path.join(txt_dir,'test_img_R.txt'), 'r').readlines()
# label_r_list = open(os.path.join(txt_dir,'test_label_R.txt'), 'r').readlines()
# img_list = img_l_list+img_r_list
# label_list = label_l_list+label_r_list
# save_path = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/test'
# for i in range(len(img_list)):
# 	img = Image.open(img_list[i].strip())
# 	label = Image.open(label_list[i].strip())
# 	print(os.path.join(save_path,'img',str(i)+'.png'))
# 	img.save(os.path.join(save_path,'img',str(i)+'.png'))
# 	label.save(os.path.join(save_path,'label',str(i)+'.png'))

