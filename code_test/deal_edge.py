#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: deal_edge.py
@time: 4/27/20 2:34 PM
@purpose:对deeplab3+的结果进行处理，使其边界的值不乱生成
'''
import cv2
import os
from tqdm import tqdm
import time

def delet_noregular_pixel(test_img,save_name):
	for i in range(384):
		for j in range(768):
			sum = test_img[i,j,0] + test_img[i,j,1] + test_img[i,j,2]
			if sum ==256:
				continue
			elif sum == 128:
				continue
			elif sum == 0:
				continue
			else:

				test_img[i, j, 0] = 0
				test_img[i, j, 1] = 0
				test_img[i, j, 2] = 0
	cv2.imwrite(save_name,test_img)
if __name__ == '__main__':

	img_path = '/media/soterea/Data_ssd/work/YYZhang/test_file/deeplabv3+/zhangyuyu1314-tensorflow-deeplab-v3-plus-master/tensorflow-deeplab-v3-plus/dataset/deal_result'
	save_path = '/media/soterea/Data_ssd/work/YYZhang/test_file/deeplabv3+/zhangyuyu1314-tensorflow-deeplab-v3-plus-master/tensorflow-deeplab-v3-plus/dataset/temp'
	print("start convert delete unuseful pixels")
	start_time = time.time()
	for name in tqdm(os.listdir(img_path)):
		img_name = os.path.join(img_path,name)
		save_name = os.path.join(save_path,name)
		test_img = cv2.imread(img_name)
		delet_noregular_pixel(test_img , save_name)
	print("conver done!",time.time()-start_time)