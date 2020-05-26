#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: choose_only_right_script.py
@time: 3/20/20 5:01 PM
@desc:
'''
import cv2
import os
from PIL import Image
raw_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/raw/right'
label_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/gtID_simple'
list_right_path = os.listdir(raw_path)
for img in list_right_path:
	img_name_label = os.path.join(label_path,img)
	print(img)
	img_label = Image.open(img_name_label)


	print(os.path.join('/media/soterea/Data_ssd/work/YYZhang/windows/deal_label/label',img))
	img_label.save(os.path.join('/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_day/color_draw/label',img))
