#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: compute_piex_object.py
@time: 3/31/20 8:19 PM
@purpose:统计各个目标所占比例,对于所有的文件夹中的图片
'''
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def count_num(img):
	r,g,b = cv2.split(img)
	r = np.array(r)
	mask_empty = (r==0)
	mask_road = (r==1)
	mask_ped = (r==2)
	mask_car = (r == 4)
	empty_arr = r[mask_empty]
	road_arr = r[mask_road]
	ped_arr = r[mask_ped]
	car_arr = r[mask_car]
	num_empty = empty_arr.size
	num_road = road_arr.size
	num_ped = ped_arr.size
	num_car = car_arr.size
	return num_empty ,num_road,num_ped,num_car

if __name__ == '__main__':
	dir_name = ['20191001','20191024','20191210','20200323']
	root_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_data'
	total_img = 0
	num_empty_total = 0
	num_road_total = 0
	num_ped_total = 0
	num_car_total = 0
	for list_name in dir_name:
		list_path = os.path.join(root_dir,list_name)
		for list_path_dir_name in os.listdir(list_path):
			path_dir = os.path.join(list_path,list_path_dir_name,'gtID_simple')
			for img_name in os.listdir(path_dir):
				img = cv2.imread(os.path.join(path_dir,img_name))
				num_empty, num_road, num_ped, num_car = count_num(img)
				total_img += 1
				num_empty_total += num_empty
				num_road_total += num_road
				num_ped_total += num_ped
				num_car_total += num_car
	total_piex = total_img*1280*320
	rate_empty = num_empty_total/total_piex
	rate_road = num_road_total/total_piex
	rate_ped = num_ped_total/total_piex
	rate_car = num_car_total/total_piex
	print("-------------总数---------------")
	print(total_piex)
	print("------------空比例---------------")
	print(rate_empty)
	print("------------路比例---------------")
	print(rate_road)
	print("------------人比例---------------")
	print(rate_ped)
	print("------------车比例---------------")
	print(rate_car)




