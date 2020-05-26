#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: count_day_night_left_right_img_number.py
@time: 4/14/20 5:49 PM
@purpose:统计训练数据/测试数据中白天，晚上，左边，右边分别数量
'''
import os
txt_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_test_txt'
path_raw  = open(os.path.join(txt_dir,'test_img_R.txt'), 'r').readlines()
number_day = 0
number_night = 0
print(len(path_raw))
for i in range(len(path_raw)):
	if 'bsd_day' in path_raw[i]:
		number_day +=1
	if 'bsd_night' in path_raw[i]:
		number_night +=1
print("----------bsd_day为-------------------------")
print(number_day)
print("----------bsd_night为-------------------------")
print(number_night)

