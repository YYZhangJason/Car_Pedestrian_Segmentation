#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: test_model_out.py
@time: 3/5/20 7:15 PM
@desc:
'''
import numpy as np
import os
import cv2
from PIL import Image
import copy
import matplotlib.pyplot as plt
import sys
#import caffe
CAFFE_HOME = '/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/' # CHANGE THIS LINE TO YOUR Caffe PATH
#CAFFE_HOME = '/home/artur/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH

sys.path.insert(0, CAFFE_HOME + 'python')

import caffe

def predic_img(net,img,res_height,res_width):
	image = copy.deepcopy(img)

	image = cv2.resize(image, (res_width, res_height), interpolation=cv2.INTER_LINEAR)

	input_blob = image.transpose((2, 0, 1))
	input_blob = input_blob[np.newaxis, ...]

	blobs = None  # ['prob', 'argMaxOut']
	out = net.forward_all(blobs=blobs, **{net.inputs[0]: input_blob})

	if 'argMaxOut' in out:
		prob = out['argMaxOut'][0]
		prediction = prob[0].astype(int)
	else:
		prob = out['prob'][0]
		prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)

	return prediction
def load_net(model,weights):
	os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
	caffe.set_mode_gpu()
	caffe.set_device(0)

	net = caffe.Net(model, weights, caffe.TEST)

	return net
def res_from_pc():
	'''set param'''
	res_height1 = 384
	res_height2 = 192
	res_width = 768
	show_img = False
	save_res = True
	record_file = '/media/soterea/Data_ssd/work/YYZhang/test_file/test_data/2020-2-15/record.csv'
	save_dir = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/2020-3-4_image_deal_with_gaocode/hight_light'

	model = '/media/soterea/Data_ssd/work/YYZhang/train_file/train_data_augmentation/sparse/deploy.prototxt'
	weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel'

	net = load_net(model,weights)

	img = cv2.imread('/media/soterea/Data_ssd/work/YYZhang/test_file/test_data/2020-2-21/image/raw/0.png')

	prediction = predic_img(net,img,res_height1,res_width)
	print (prediction)
	plt.figure("demo")
	plt.imshow(prediction)
	plt.show()
if __name__ == '__main__':
	res_from_pc()