#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: compute_no_object_number.py
@time: 4/9/20 10:57 AM
@purpose:统计所有训练图像中没有行人或者汽车的图片
'''
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def count_num(img):
	r,g,b = cv2.split(img)
	r = np.array(r)
	mask_ped = (r==2)
	mask_car = (r == 4)
	ped_arr = r[mask_ped]
	car_arr = r[mask_car]
	num_ped = ped_arr.size
	num_car = car_arr.size
	return  num_ped,num_car

if __name__ == '__main__':
	dir_name = ['20191001','20191024','20191210','20200323']
	root_dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data'
	total_img = 0
	no_ped_total = 0
	no_car_total = 0
	no_ped_car_total = 0
	for list_name in dir_name:
		list_path = os.path.join(root_dir,list_name)
		for list_path_dir_name in os.listdir(list_path):
			path_dir = os.path.join(list_path,list_path_dir_name,'gtID_simple')
			for img_name in os.listdir(path_dir):
				img = cv2.imread(os.path.join(path_dir,img_name))
				num_ped, num_car = count_num(img)
				total_img += 1
				if num_ped == 0:
					no_ped_total +=1
				if num_car == 0:
					no_car_total +=1
				if num_car ==0 and num_ped ==0:
					no_ped_car_total +=1
	print("-------------总数---------------")
	print(total_img)
	print("------------没有人---------------")
	print(no_ped_total)
	print("------------没有车---------------")
	print(no_car_total)
	print("------------同时没有人、车---------------")
	print(no_ped_car_total)





