#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: test.py
@time: 2/25/20 12:41 PM
@desc:
'''
import time
import numpy as np
from datetime import datetime
# import cv2
# import matplotlib.pyplot as plt
#
# img = cv2.imread('/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/20191024/bsd_day/raw_img/left/2019-08-22-08-51-03_800_left.png')
# img = cv2.resize(img,(768,384),interpolation=cv2.INTER_LINEAR)
# cv2.imwrite("2.png",img)
#
# plt.figure("demo")
# plt.imshow(img)
# plt.show()
import cv2
import matplotlib.pyplot as plt
# width = 768
# bsd_area = np.asarray([[(230, 42), (66, 372), (868, 410), (764, 96)]],
# 					  dtype=np.int)
# bsd_area = (bsd_area*(width/1024.0)).astype(np.int)
# print(bsd_area)
# img = cv2.imread('/media/soterea/Data_ssd/work/YYZhang/test_file/test_data/2020-2-21/image/raw/0.png')
#
# for pts in bsd_area:
# 	pts = np.array(pts, np.int32)
# 	pts = pts.reshape((-1, 1, 2))
# 	cv2.polylines(img, [pts], True, (0, 255, 0), 2)
#
# plt.figure("demo")
# plt.imshow(img)
# plt.show()
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import numpy as np



img = Image.open('/media/soterea/Data_ssd/work/YYZhang/test_file/deeplabv3+/zhangyuyu1314-tensorflow-deeplab-v3-plus-master/tensorflow-deeplab-v3-plus/dataset/save_deep/gtID_simple/2019-06-01-02-53-03_021400_right.png')

plt.figure()
plt.imshow(img)
plt.show()
# # import sys
# CAFFE_HOME = '/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/python'
#
# sys.path.insert(0, CAFFE_HOME)
#
# import caffe
#
# print(sys.path)
# import os
#
# img_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_data/20200323/bsd_night/raw_img'
# for subdir, dirs, files in os.walk(img_dir):
# 	print("++++++++++++subdir+++++++++")
# 	print (subdir)
# 	print("++++++++++++dirs+++++++++")
# 	print(dirs)
# 	print("++++++++++++files+++++++++")
# 	print(files)