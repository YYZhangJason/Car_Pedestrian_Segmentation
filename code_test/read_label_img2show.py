#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: read_label_img2show.py
@time: 3/11/20 3:06 PM
@desc:
'''
import cv2
import numpy as np
#print label_img
from PIL import  Image
import matplotlib.pyplot as plt
im = Image.open('/media/soterea/Data_ssd/work/YYZhang/test_file/convert_result/gtID_simple/2019-11-12-10-13-52_16_left.png')
print im.size
print (np.array(im))
plt.figure('test')
plt.imshow(im)
plt.show()
width = im.size[0]
height = im.size[1]

