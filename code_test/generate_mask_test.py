#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: generate_mask_test.py
@time: 5/7/20 2:40 PM
@purpose:测试生成的mask
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np


class my_class(object):

	def __init__(self):

		self.mask = np.zeros((10,10))
		self.bsd_area = np.asarray([[(0,0),(3,3),(0,3),(3,0)]])
		for points in self.bsd_area:
			cv2.fillPoly(self.mask,[np.array(points)],1)
	def use_param(self):
		print(self.mask)
		plt.imshow(self.mask)
		plt.show()

P = my_class()
P.use_param()