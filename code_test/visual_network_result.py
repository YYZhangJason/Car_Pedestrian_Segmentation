#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: visual_network.py
@time: 3/5/20 6:39 PM
@desc:
'''
# !/usr/bin/env python

from __future__ import print_function
import sys
import numpy as np
import os, glob
import cv2
# myMod
import sys

CAFFE_HOME = '/media/artur/Dataset/work/02_FrameWork/caffe-jacinto/'  # CHANGE THIS LINE TO YOUR Caffe PATH
# CAFFE_HOME = '/home/artur/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_HOME + 'python')
import caffe
# import lmdb
from PIL import Image
import argparse
import random
import shutil
import imageio
import math
import time
import caffe.proto.caffe_pb2 as caffe_pb2
import matplotlib.pyplot as plt
def visual_conv_data(data):
	#输入的数据格式：ndarray
	#尺寸：(n,height,width) or (n,height,width,3)

	#normalization on input data
	#data = (data-data.min())/(data.max-data.min())
	n = int(np.ceil(np.sqrt(data.shape[0])))

	padding_between_img_and_img = (((0,n**2-data.shape[0]),
									(0,1),(0,1))+((0,0),)*(data.ndim-3))
	data = np.pad(data,padding_between_img_and_img,mode = 'constant',constant_values=1)
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	fig, ax = plt.subplots()
	im = data
	ax.imshow(im, aspect='equal')
	plt.axis('off')
	height, width = im.shape
	fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
	plt.margins(0, 0)
	plt.show()
	plt.savefig('test.png',dpi=300)


def check_network():
	os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
	# caffe.set_mode_gpu()
	# caffe.set_device(0)
	caffe.set_mode_cpu()
	model = '/media/soterea/Data_ssd/work/YYZhang/train_file/2020/5/train_fusion_lower_feature/sparse/deploy.prototxt'
	weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020/5/sparse/add_conv2_feature_to_combine/BSD_train_sparse_d-n_iter_60000.caffemodel'
	# weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3.30-4.10/3.30-only-right-day-night-newdata/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel'

	intput_img = '/media/soterea/Data_ssd/work/YYZhang/data/test_data/testfrom_Litao/2020-05-09-09-50-00/right/2020-05-09-09-50-00_33_right.png'
	net = caffe.Net(model, weights, caffe.TEST)
	image = cv2.imread(intput_img)



	image = cv2.resize(image, (768, 384), interpolation=cv2.INTER_LINEAR)

	input_blob = image.transpose((2, 0, 1))
	input_blob = input_blob[np.newaxis, ...]

	blobs = None  # ['prob', 'argMaxOut']
	out = net.forward_all(blobs=blobs, **{net.inputs[0]: input_blob})

	for layer_name ,blob in net.blobs.iteritems():# # 查看每一层的输出纬度、大小
		print(layer_name+':'+ str(blob.data.shape))
	filters = net.blobs['out_deconv_final_up8'].data[0]
	print(filters.shape)
	prob = np.subtract(filters, np.max(filters, axis=0)[np.newaxis, :, ...])
	prob = np.exp(prob)
	softmax_value = np.divide(prob, np.sum(prob, axis=0)[np.newaxis, :, ...])
	pred_ped = softmax_value[2, :, :]
	# print(pred_ped.shape)
	# print(pred_ped)
	# for i in range(5):
	# 	mask = (filters[i,:,:]<15)
	# 	filters[i, :, :][mask] = 0

	# print(mask.shape)
	#
	#
	# # # print(filters.shape)
	result = np.argmax(filters,axis=0)

	mask = (result==2)&(pred_ped<0.8)
	result[mask] =0
	#pred_argmax = np.argmax(pred_batch, axis=1)
	plt.imshow(result)
	plt.show()
	# print(result.shape)
	#visual_conv_data(filters)
if __name__ == "__main__":
	check_network()
