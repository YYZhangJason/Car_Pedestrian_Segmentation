#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: Loss_scooter_as_Ped.py
@time: 4/14/20 2:04 PM
@purpose:
'''
from __future__ import division

import sys
CAFFE_HOME = '/media/soterea/Data_ssd/04_Caffe_Folder/caffe-jacinto/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_HOME + 'python')

import caffe
import numpy as np
from PIL import Image
import os
import cv2
import argparse
import copy
import time
import cupy as cp
import sys

class Balance_loss(caffe.Layer):
	"""
	Compute the Sofamax in the same manner as the C++ Softmax Layer
	to demonstrate the class interface for developing layers in Python.
	"""


	def setup(self, bottom, top):
		self.debug = False
		if len(bottom) != 2:
			raise Exception("Softmaxwithloss Layer Needs two blob as input.")
		if len(top) != 1:
			raise Exception("Softmaxwithloss Layer takes a single blob as output.")
		params = eval(self.param_str)
		self.has_ignore_label = params.get('has_ignore_label',False)
		self.ignore_label = params.get('ignore_label',255)
		self.weight_dic = eval(params.get('weight_dic'))
		self.alpha = 0.25
		self.gamma = 2
		print("+++++++++++++++++self.ignore_label++++++++++++++++++++++++++++++++++++++++")
		print(self.ignore_label)
		print(self.has_ignore_label)
		self.softmax_axis = 1
	def reshape(self, bottom, top):
		outer_num = bottom[0].data.shape[0]
		channel = bottom[0].data.shape[1]
		height,width = bottom[0].data.shape[2],bottom[0].data.shape[3]
		self.index_0 = cp.arange(outer_num).reshape([outer_num,1,1,1])
		self.index_3,self.index_2 = cp.meshgrid(cp.arange(width),cp.arange(height))

		top[0].reshape(1)

	def forward(self, bottom, top):
		self.label = cp.asarray(copy.deepcopy(bottom[1].data),cp.uint8)
		prob = cp.asarray(copy.deepcopy(bottom[0].data),cp.float64)
		prob = cp.subtract(prob,cp.max(prob,axis=1)[:,cp.newaxis,...])
		prob = cp.exp(prob)
		self.softmax = cp.divide(prob,cp.sum(prob,axis=1)[:,cp.newaxis,...])

		## mask
		self.weight_mask = cp.ones_like(self.label, cp.float64)
		for weight_id in self.weight_dic:
			self.weight_mask[self.label == weight_id] = self.weight_dic[weight_id]

		if self.has_ignore_label:
			self.weight_mask[self.label == self.ignore_label] = 0
		# num_total = 15422668800
		# empty_num = 3679002314
		# road_num = 10565335603
		# ped_num = 99066996
		# car_num = 995347874
		self.label[self.label == 3] = 2
		# w_empty = float((num_total-empty_num)/num_total)
		# w_road = float((num_total-road_num)/num_total)
		# w_ped = float((num_total-ped_num)/num_total)
		# w_car = float((num_total-car_num)/num_total)
		# print(w_empty)
		# print(w_road)
		# print(w_ped)
		# print(w_car)
		# empty:0.3
		# road:0.25

		self.weight_mask[self.label == 0] = 0.3
		self.weight_mask[self.label == 1] = 0.25
		# self.weight_mask[self.label == 2] = w_ped
		# self.weight_mask[self.label == 4] = w_car



		compute_count = self.weight_mask[self.weight_mask != 0].size

		## nomalize mask
		self.weight_mask = cp.divide(self.weight_mask, cp.divide(cp.sum(self.weight_mask), compute_count))


		## compute loss
		prob_compute_matrix = copy.deepcopy(self.softmax[self.index_0,self.label,self.index_2,self.index_3])
		prob_compute_matrix[prob_compute_matrix < (1e-10)] = 1e-10
		loss = - cp.divide(cp.sum(cp.multiply(cp.log(prob_compute_matrix),self.weight_mask)),compute_count)

		loss = cp.asnumpy(loss)
		top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):

		if propagate_down[0]:

			bottom_diff = copy.deepcopy(self.softmax)
			bottom_diff[self.index_0,self.label,self.index_2,self.index_3] -=1
			bottom_diff = cp.multiply(bottom_diff,self.weight_mask)
			bottom_diff = cp.divide(bottom_diff,self.weight_mask[self.weight_mask != 0].size)

			bottom_diff = cp.asnumpy(bottom_diff)
			bottom[0].diff[...]=bottom_diff
class Under_mask_loss(caffe.Layer):
	"""
	Compute the Sofamax in the same manner as the C++ Softmax Layer
	to demonstrate the class interface for developing layers in Python.
	"""
	def setup(self, bottom, top):
		self.debug = False
		if len(bottom) != 3:
			raise Exception("Softmaxwithloss Layer Needs two blob as input.")
		if len(top) != 1:
			raise Exception("Softmaxwithloss Layer takes a single blob as output.")
		params = eval(self.param_str)
		self.has_ignore_label = params.get('has_ignore_label',False)
		self.ignore_label = params.get('ignore_label',255)
		self.weight_dic = eval(params.get('weight_dic'))
		self.alpha = 0.25
		self.gamma = 2
		print("+++++++++++++++++self.ignore_label++++++++++++++++++++++++++++++++++++++++")
		print(self.ignore_label)
		print(self.has_ignore_label)
		self.softmax_axis = 1
	def reshape(self, bottom, top):
		outer_num = bottom[0].data.shape[0]
		channel = bottom[0].data.shape[1]
		height,width = bottom[0].data.shape[2],bottom[0].data.shape[3]
		self.index_0 = cp.arange(outer_num).reshape([outer_num,1,1,1])
		self.index_3,self.index_2 = cp.meshgrid(cp.arange(width),cp.arange(height))

		top[0].reshape(1)

	def forward(self, bottom, top):
		self.label = cp.asarray(copy.deepcopy(bottom[1].data),cp.uint8)
		self.mask = cp.asarray(copy.deepcopy(bottom[2].data),cp.uint8)


		prob = cp.asarray(copy.deepcopy(bottom[0].data),cp.float64)
		prob = cp.subtract(prob,cp.max(prob,axis=1)[:,cp.newaxis,...])
		prob = cp.exp(prob)
		self.softmax = cp.divide(prob,cp.sum(prob,axis=1)[:,cp.newaxis,...])

		## mask
		self.weight_mask = cp.ones_like(self.label, cp.float64)
		for weight_id in self.weight_dic:
			self.weight_mask[self.label == weight_id] = self.weight_dic[weight_id]

		if self.has_ignore_label:
			self.weight_mask[self.label == self.ignore_label] = 0
		# num_total = 15422668800
		# empty_num = 3679002314
		# road_num = 10565335603
		# ped_num = 99066996
		# car_num = 995347874

		self.label[self.label == 3] = 2
		#self.mask[self.label != 2] = 1

		# w_empty = float((num_total-empty_num)/num_total)
		# w_road = float((num_total-road_num)/num_total)
		# w_ped = float((num_total-ped_num)/num_total)
		# w_car = float((num_total-car_num)/num_total)
		# print(w_empty)
		# print(w_road)
		# print(w_ped)
		# print(w_car)
		# empty:0.3
		# road:0.25



		#self.mask[self.label!=2] = 1





		self.weight_mask[self.label == 0] = 0.3
		self.weight_mask[self.label == 1] = 0.25
		self.weight_mask[self.mask == 0] = 0.1

		# self.weight_mask[self.label == 2] = w_ped
		# self.weight_mask[self.label == 4] = w_car



		compute_count = self.weight_mask[self.weight_mask != 0].size

		## nomalize mask
		self.weight_mask = cp.divide(self.weight_mask, cp.divide(cp.sum(self.weight_mask), compute_count))


		## compute loss
		prob_compute_matrix = copy.deepcopy(self.softmax[self.index_0,self.label,self.index_2,self.index_3])
		prob_compute_matrix[prob_compute_matrix < (1e-10)] = 1e-10
		loss = - cp.divide(cp.sum(cp.multiply(cp.log(prob_compute_matrix),self.weight_mask)),compute_count)

		loss = cp.asnumpy(loss)
		top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):

		if propagate_down[0]:

			bottom_diff = copy.deepcopy(self.softmax)
			bottom_diff[self.index_0,self.label,self.index_2,self.index_3] -=1
			bottom_diff = cp.multiply(bottom_diff,self.weight_mask)
			bottom_diff = cp.divide(bottom_diff,self.weight_mask[self.weight_mask != 0].size)

			bottom_diff = cp.asnumpy(bottom_diff)
			bottom[0].diff[...]=bottom_diff
class Mask_loss(caffe.Layer):
	"""
	Compute the Sofamax in the same manner as the C++ Softmax Layer
	to demonstrate the class interface for developing layers in Python.
	"""


	def setup(self, bottom, top):
		self.debug = False
		if len(bottom) != 2:
			raise Exception("Softmaxwithloss Layer Needs two blob as input.")
		if len(top) != 1:
			raise Exception("Softmaxwithloss Layer takes a single blob as output.")
		params = eval(self.param_str)
		self.has_ignore_label = params.get('has_ignore_label',False)
		self.ignore_label = params.get('ignore_label',255)
		self.weight_dic = eval(params.get('weight_dic'))
		self.alpha = 0.25
		self.gamma = 2
		print("+++++++++++++++++self.ignore_label++++++++++++++++++++++++++++++++++++++++")
		print(self.ignore_label)
		print(self.has_ignore_label)
		self.softmax_axis = 1
	def reshape(self, bottom, top):
		outer_num = bottom[0].data.shape[0]
		channel = bottom[0].data.shape[1]
		height,width = bottom[0].data.shape[2],bottom[0].data.shape[3]
		self.index_0 = cp.arange(outer_num).reshape([outer_num,1,1,1])
		self.index_3,self.index_2 = cp.meshgrid(cp.arange(width),cp.arange(height))

		top[0].reshape(1)

	def forward(self, bottom, top):
		self.label = cp.asarray(copy.deepcopy(bottom[1].data),cp.uint8)
		prob = cp.asarray(copy.deepcopy(bottom[0].data),cp.float64)
		prob = cp.subtract(prob,cp.max(prob,axis=1)[:,cp.newaxis,...])
		prob = cp.exp(prob)
		self.softmax = cp.divide(prob,cp.sum(prob,axis=1)[:,cp.newaxis,...])

		## mask
		self.weight_mask = cp.ones_like(self.label, cp.float64)
		for weight_id in self.weight_dic:
			self.weight_mask[self.label == weight_id] = self.weight_dic[weight_id]

		if self.has_ignore_label:
			self.weight_mask[self.label == self.ignore_label] = 0
		# num_total = 15422668800
		# empty_num = 3679002314
		# road_num = 10565335603
		# ped_num = 99066996
		# car_num = 995347874
		#self.label[self.label == 3] = 2
		# w_empty = float((num_total-empty_num)/num_total)
		# w_road = float((num_total-road_num)/num_total)
		# w_ped = float((num_total-ped_num)/num_total)
		# w_car = float((num_total-car_num)/num_total)
		# print(w_empty)
		# print(w_road)
		# print(w_ped)
		# print(w_car)
		# empty:0.3
		# road:0.25

		self.weight_mask[self.label == 0] = 0.3
		self.weight_mask[self.label == 1] = 0.25
		self.weight_mask[self.label == 3] = 0.5

		# self.weight_mask[self.label == 2] = w_ped
		# self.weight_mask[self.label == 4] = w_car



		compute_count = self.weight_mask[self.weight_mask != 0].size


		## nomalize mask
		self.weight_mask = cp.divide(self.weight_mask, cp.divide(cp.sum(self.weight_mask), compute_count))


		## compute loss
		prob_compute_matrix = copy.deepcopy(self.softmax[self.index_0,self.label,self.index_2,self.index_3])
		prob_compute_matrix[prob_compute_matrix < (1e-10)] = 1e-10
		loss = - cp.divide(cp.sum(cp.multiply(cp.log(prob_compute_matrix),self.weight_mask)),compute_count)

		loss = cp.asnumpy(loss)
		top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):

		if propagate_down[0]:

			bottom_diff = copy.deepcopy(self.softmax)
			bottom_diff[self.index_0,self.label,self.index_2,self.index_3] -=1
			bottom_diff = cp.multiply(bottom_diff,self.weight_mask)
			bottom_diff = cp.divide(bottom_diff,self.weight_mask[self.weight_mask != 0].size)

			bottom_diff = cp.asnumpy(bottom_diff)
			bottom[0].diff[...]=bottom_diff
class loss_baseline_scooter_as_ped(caffe.Layer):
	"""
	Compute the Sofamax in the same manner as the C++ Softmax Layer
	to demonstrate the class interface for developing layers in Python.
	"""


	def setup(self, bottom, top):
		self.debug = False
		if len(bottom) != 2:
			raise Exception("Softmaxwithloss Layer Needs two blob as input.")
		if len(top) != 1:
			raise Exception("Softmaxwithloss Layer takes a single blob as output.")
		params = eval(self.param_str)
		self.has_ignore_label = params.get('has_ignore_label',False)
		self.ignore_label = params.get('ignore_label',255)
		self.weight_dic = eval(params.get('weight_dic'))
		print("+++++++++++++++++self.ignore_label++++++++++++++++++++++++++++++++++++++++")
		print(self.ignore_label)
		print(self.has_ignore_label)
		self.softmax_axis = 1
	def reshape(self, bottom, top):
		outer_num = bottom[0].data.shape[0]
		channel = bottom[0].data.shape[1]
		height,width = bottom[0].data.shape[2],bottom[0].data.shape[3]
		self.index_0 = cp.arange(outer_num).reshape([outer_num,1,1,1])
		self.index_3,self.index_2 = cp.meshgrid(cp.arange(width),cp.arange(height))

		top[0].reshape(1)

	def forward(self, bottom, top):
		self.label = cp.asarray(copy.deepcopy(bottom[1].data),cp.uint8)
		prob = cp.asarray(copy.deepcopy(bottom[0].data),cp.float64)
		prob = cp.subtract(prob,cp.max(prob,axis=1)[:,cp.newaxis,...])
		prob = cp.exp(prob)
		self.softmax = cp.divide(prob,cp.sum(prob,axis=1)[:,cp.newaxis,...])

		## mask
		self.weight_mask = cp.ones_like(self.label, cp.float64)
		for weight_id in self.weight_dic:
			self.weight_mask[self.label == weight_id] = self.weight_dic[weight_id]

		if self.has_ignore_label:
			self.weight_mask[self.label == self.ignore_label] = 0
		# self.weight_mask[self.label == 0] = 0.3
		# self.weight_mask[self.label == 1] = 0.25
		# self.weight_mask[self.label == 2] = 5
		# self.weight_mask[self.label == 4] = 2
		self.label[self.label == 3] = 2


		compute_count = self.weight_mask[self.weight_mask != 0].size

		## nomalize mask
		self.weight_mask = cp.divide(self.weight_mask, cp.divide(cp.sum(self.weight_mask), compute_count))


		## compute loss
		prob_compute_matrix = copy.deepcopy(self.softmax[self.index_0,self.label,self.index_2,self.index_3])
		prob_compute_matrix[prob_compute_matrix < (1e-10)] = 1e-10

		loss = - cp.divide(cp.sum(cp.multiply(cp.log(prob_compute_matrix),self.weight_mask)),compute_count)

		loss = cp.asnumpy(loss)
		top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):

		if propagate_down[0]:

			bottom_diff = copy.deepcopy(self.softmax)
			bottom_diff[self.index_0,self.label,self.index_2,self.index_3] -=1
			bottom_diff = cp.multiply(bottom_diff,self.weight_mask)
			bottom_diff = cp.divide(bottom_diff,self.weight_mask[self.weight_mask != 0].size)

			bottom_diff = cp.asnumpy(bottom_diff)
			bottom[0].diff[...]=bottom_diff
class Penalty_loss(caffe.Layer):
	"""
	Compute the Sofamax in the same manner as the C++ Softmax Layer
	to demonstrate the class interface for developing layers in Python.
	"""


	def setup(self, bottom, top):
		self.debug = False
		if len(bottom) != 2:
			raise Exception("Softmaxwithloss Layer Needs two blob as input.")
		if len(top) != 1:
			raise Exception("Softmaxwithloss Layer takes a single blob as output.")
		params = eval(self.param_str)
		self.has_ignore_label = params.get('has_ignore_label',False)
		self.ignore_label = params.get('ignore_label',255)
		self.weight_dic = eval(params.get('weight_dic'))
		self.alpha = 0.25
		self.gamma = 2
		self.softmax_axis = 1
		print("+++++++++++++++++++Penalty_5x+++++++++++++++++++++")
	def reshape(self, bottom, top):
		outer_num = bottom[0].data.shape[0]
		channel = bottom[0].data.shape[1]
		height,width = bottom[0].data.shape[2],bottom[0].data.shape[3]
		self.index_0 = cp.arange(outer_num).reshape([outer_num,1,1,1])
		self.index_3,self.index_2 = cp.meshgrid(cp.arange(width),cp.arange(height))

		top[0].reshape(1)

	def forward(self, bottom, top):
		self.label = cp.asarray(copy.deepcopy(bottom[1].data),cp.uint8)
		prob = cp.asarray(copy.deepcopy(bottom[0].data),cp.float64)
		result = np.argmax(prob, axis=1)[:,cp.newaxis,...]
		prob = cp.subtract(prob,cp.max(prob,axis=1)[:,cp.newaxis,...])
		prob = cp.exp(prob)
		self.softmax = cp.divide(prob,cp.sum(prob,axis=1)[:,cp.newaxis,...])



		## mask
		self.weight_mask = cp.ones_like(self.label, cp.float64)
		for weight_id in self.weight_dic:
			self.weight_mask[self.label == weight_id] = self.weight_dic[weight_id]

		if self.has_ignore_label:
			self.weight_mask[self.label == self.ignore_label] = 0
		# num_total = 15422668800
		# empty_num = 3679002314
		# road_num = 10565335603
		# ped_num = 99066996
		# car_num = 995347874
		self.label[self.label == 3] = 2

		#self.penalty_mask = (self.label != result)&((self.label==2)|(self.label==4))#这边加惩罚是将预测结果和label不相等，即预测错误的时候，如果标签是2或者4，就给其loss加大权重，相当于加了一个惩罚

		self.penalty_mask = (self.label != result)&((result==2)|(result==4))#这边加惩罚是将预测结果和label不相等，即预测错误的时候，如果预测的结果是2或者4，即不是2，4类别的被预测成2，4类别，给其一个惩罚

		# w_empty = float((num_total-empty_num)/num_total)
		# w_road = float((num_total-road_num)/num_total)
		# w_ped = float((num_total-ped_num)/num_total)
		# w_car = float((num_total-car_num)/num_total)
		# print(w_empty)
		# print(w_road)
		# print(w_ped)
		# print(w_car)
		# empty:0.3
		# road:0.25

		self.weight_mask[self.label == 0] = 0.3
		self.weight_mask[self.label == 1] = 0.25
		self.weight_mask[self.penalty_mask] = self.weight_mask[self.penalty_mask]*2

		# self.weight_mask[self.label == 2] = w_ped
		# self.weight_mask[self.label == 4] = w_car



		compute_count = self.weight_mask[self.weight_mask != 0].size

		## nomalize mask
		self.weight_mask = cp.divide(self.weight_mask, cp.divide(cp.sum(self.weight_mask), compute_count))


		## compute loss
		prob_compute_matrix = copy.deepcopy(self.softmax[self.index_0,self.label,self.index_2,self.index_3])
		prob_compute_matrix[prob_compute_matrix < (1e-10)] = 1e-10
		loss = - cp.divide(cp.sum(cp.multiply(cp.log(prob_compute_matrix),self.weight_mask)),compute_count)

		loss = cp.asnumpy(loss)
		top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):

		if propagate_down[0]:

			bottom_diff = copy.deepcopy(self.softmax)
			bottom_diff[self.index_0,self.label,self.index_2,self.index_3] -=1
			bottom_diff = cp.multiply(bottom_diff,self.weight_mask)
			bottom_diff = cp.divide(bottom_diff,self.weight_mask[self.weight_mask != 0].size)

			bottom_diff = cp.asnumpy(bottom_diff)
			bottom[0].diff[...]=bottom_diff
