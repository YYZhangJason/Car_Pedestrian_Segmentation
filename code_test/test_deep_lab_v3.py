#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: test_deep_lab_v3.py
@time: 4/26/20 4:59 PM
@purpose:测试deeplabv3+结果
'''
import cv2
import os
from PIL import Image
#deal predict result
pred_result_path = '/media/soterea/Data_ssd/work/YYZhang/test_file/deeplabv3+/zhangyuyu1314-tensorflow-deeplab-v3-plus-master/tensorflow-deeplab-v3-plus/dataset/inference_output'
save_dir = '/media/soterea/Data_ssd/work/YYZhang/test_file/deeplabv3+/zhangyuyu1314-tensorflow-deeplab-v3-plus-master/tensorflow-deeplab-v3-plus/dataset/deal_result'
for name in os.listdir(pred_result_path):
	img_path = os.path.join(pred_result_path,name)
	img = Image.open(img_path)

	img = cv2.imread(img_path)
	img = img[10:134,10:506]
	img_ = cv2.resize(img,(768,384),interpolation=cv2.INTER_NEAREST)

	save_path = os.path.join(save_dir,name)
	cv2.imwrite(save_path,img_)