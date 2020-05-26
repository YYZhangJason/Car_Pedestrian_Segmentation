#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: BSDLayer_only_right_side.py
@time: 3/12/20 2:28 PM
@desc:
'''
#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: BSDLayer_new.py
@time: 2/20/20 7:12 PM
@desc:
'''
import sys
CAFFE_HOME = '/media/artur/Dataset/work/02_FrameWork/caffe-jacinto/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_HOME + 'python')

import caffe
import numpy as np
from PIL import Image
import os
import cv2
import Automold as am
import Helpers as hp

class BSDLayer_train_data_augmentation_mixed(caffe.Layer):

	"""
	Load BSD single images
	shuffle left and right, 1/4 possibility random filp,
	mix ped-only & no-ped
	under txt_dir has txt file list for training

    """

	def setup(self, bottom, top):
		"""
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
		''' config '''
		params = eval(self.param_str)
		self.split = params['split']
		self.seed = params.get('seed')
		self.batch_size = params.get('batch_size')
		self.resize_h = params.get('resize_h')
		self.resize_w = params.get('resize_w')
		self.random_crop = params.get('random_crop')
		self.crop_h = params.get('crop_h')
		self.crop_w = params.get('crop_w')
		txt_dir = params.get('txt_dir')
		self.bsd_area = np.asarray([[(150,20),(46,450),(960,460),(800,66)]],dtype=np.int)
		self.mask = np.zeros((384,768))
		self.bsd_area =(self.bsd_area*(768/1024)).astype(np.int)
		for points in self.bsd_area:
			cv2.fillPoly(self.mask,[np.array(points)],1)



		# three tops: data, semantic, geometric
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")


		img_ped_list = open(os.path.join(txt_dir,'train_img_L.txt'), 'r').readlines()
		label_ped_list = open(os.path.join(txt_dir,'train_label_L.txt'), 'r').readlines()
		img_noped_list = open(os.path.join(txt_dir,'train_img_R.txt'), 'r').readlines()
		label_noped_list = open(os.path.join(txt_dir,'train_label_R.txt'), 'r').readlines()

		self.train_img_list = img_ped_list + img_noped_list
		self.train_label_list = label_ped_list + label_noped_list
		print("+++++++++++++train_img_list")
		print(len(self.train_img_list))
		print(len(self.train_label_list))
		np.random.seed(self.seed)
		state = np.random.get_state()
		np.random.shuffle(self.train_img_list)
		np.random.set_state(state)
		np.random.shuffle(self.train_label_list)
		self.id = 0


	def reshape(self, bottom, top):
		# load image + label image pair
		self.data = []
		self.label = []
		for i in range(self.batch_size):

			data_,label_ = self.load_image_label()
			self.data.append(data_)
			self.label.append(label_)

		state = np.random.get_state()
		np.random.shuffle(self.data)
		np.random.set_state(state)
		np.random.shuffle(self.label)

		self.data = np.array(self.data)
		self.label = np.array(self.label)
		# print  'data.shape:', self.data.shape,  'label.shape:', self.label.shape
		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(*self.data.shape)
		top[1].reshape(*self.label.shape)

	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label


	def backward(self, top, propagate_down, bottom):
		pass

	def load_image_label(self):
		"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

		if (self.id +1) >= len(self.train_img_list):
			state = np.random.get_state()
			np.random.shuffle(self.train_img_list)
			np.random.set_state(state)
			np.random.shuffle(self.train_label_list)
			self.id = 0


		#img_l_path = self.train_img_list[self.id%len(self.train_img_list)].strip()
		#label_l_path = self.train_label_list[self.id%len(self.train_img_list)].strip()
		img_r_path = self.train_img_list[self.id].strip()
		label_r_path = self.train_label_list[self.id].strip()
		#print(img_r_path)

		# if os.path.basename(img_l_path) != os.path.basename(label_l_path):
		# 	print('Warning: Img ', os.path.basename(img_l_path), 'Label ', os.path.basename(label_l_path), 'DIFF')
		if os.path.basename(img_r_path.strip('.png')) != os.path.basename(label_r_path.strip('.png')):
			print('Warning: Img ', os.path.basename(img_r_path), 'Label ', os.path.basename(label_r_path), 'DIFF')

		#img_l = cv2.imread(img_l_path)
		img_r = cv2.imread(img_r_path)

		#label_l = Image.open(label_l_path)
		#label_l = np.array(label_l, dtype=np.uint8)
		label_r = Image.open(label_r_path)
		label_r = np.array(label_r, dtype=np.uint8)


		# print 'label_l_path',label_l_path
		# print 'label_r_path',label_r_path
		# print 'label_l.size',label_l.shape
		# print 'label_r.size', label_r.shape

		##filp
		# if np.random.randint(4) == 0:
		# 	img_l = np.flip(img_l,axis=1)
		# 	label_l = np.flip(label_l,axis=1)
		# if np.random.randint(4) == 0:
		# 	img_r = np.flip(img_r,axis=1)
		# 	label_r = np.flip(label_r,axis=1)


		## rotate img_r
		# img_r = np.rot90(img_r, 2)
		# label_r = np.rot90(label_r, 2)

		## merge
		img_all = np.array(img_r).astype(np.uint8)
		#img_all = np.vstack((img_l,img_r)).astype(np.uint8)
		#label_all =np.vstack((label_l,label_r)).astype(np.uint8)
		label_all = label_r

		## rotate img_all
		# img_all = np.rot90(img_all, 2)
		# label_all = np.rot90(label_all, 2)

		# resize
		img_all = cv2.resize(img_all, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
		label_all = Image.fromarray(label_all, 'P')
		label_all = label_all.resize([self.resize_w, self.resize_h], Image.NEAREST)
		label_all = np.array(label_all, dtype=np.uint8)

		## crop
		if self.random_crop:
			img_all,label_all = self.random_crop_data(img_all,label_all)

		# ## test
		# cv2.imshow('img',img_all)
		# cv2.imshow('label',label_all)
		# sys.stdout.flush()
		# sys.stdout.write('shape'+ str(img_all.shape))
		# cv2.waitKey()
		# ##
		self.id +=1
		img_all = img_all.transpose((2, 0, 1))
		label_all = label_all[np.newaxis, ...]


		return img_all,label_all

	def random_crop_data(self,img,label):
		img = np.array(img, dtype=np.uint8)

		max_width = img.shape[1]
		max_height = img.shape[0]

		if ((max_width - self.crop_w) < 0) or ((max_height - self.crop_h)<0):
			print 'random crop size is larger than image size!'

		rand_width = np.random.randint(0, max_width - self.crop_w)
		rand_height = np.random.randint(0, max_height - self.crop_h)

		img = img[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		label = label[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		return img,label
class BSD_DataLayer_train_under_mask(caffe.Layer):

	"""
	Load BSD single images
	shuffle left and right, 1/4 possibility random filp,
	mix ped-only & no-ped
	under txt_dir has txt file list for training

    """

	def setup(self, bottom, top):
		"""
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
		''' config '''
		params = eval(self.param_str)
		self.split = params['split']
		self.seed = params.get('seed')
		self.batch_size = params.get('batch_size')
		self.resize_h = params.get('resize_h')
		self.resize_w = params.get('resize_w')
		self.random_crop = params.get('random_crop')
		self.crop_h = params.get('crop_h')
		self.crop_w = params.get('crop_w')
		txt_dir = params.get('txt_dir')
		# self.bsd_area = np.asarray([[(0,140),(0,384),(768,384),(768,104)]],dtype=np.int)
		# self.mask = np.zeros((384,768))
		# for points in self.bsd_area:
		# 	cv2.fillPoly(self.mask,[np.array(points)],1)#Artur说的mask方法

		self.bsd_area = np.asarray([[(150, 20), (46, 450), (960, 460), (800, 66)]], dtype=np.int)
		self.mask = np.zeros((384, 768))
		self.bsd_area = (self.bsd_area * 0.75).astype(np.int)
		print(self.bsd_area)
		for points in self.bsd_area:
			cv2.fillPoly(self.mask, [np.array(points)], 1)



		# three tops: data, semantic, geometric
		if len(top) != 3:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")


		img_ped_list = open(os.path.join(txt_dir,'train_img_L.txt'), 'r').readlines()
		label_ped_list = open(os.path.join(txt_dir,'train_label_L.txt'), 'r').readlines()
		img_noped_list = open(os.path.join(txt_dir,'train_img_R.txt'), 'r').readlines()
		label_noped_list = open(os.path.join(txt_dir,'train_label_R.txt'), 'r').readlines()

		self.train_img_list = img_ped_list + img_noped_list
		self.train_label_list = label_ped_list + label_noped_list
		print("+++++++++++++train_img_list")
		print(len(self.train_img_list))
		print(len(self.train_label_list))
		np.random.seed(self.seed)
		state = np.random.get_state()
		np.random.shuffle(self.train_img_list)
		np.random.set_state(state)
		np.random.shuffle(self.train_label_list)
		self.id = 0


	def reshape(self, bottom, top):
		# load image + label image pair
		self.data = []
		self.label = []
		self.mask_ = []
		for i in range(self.batch_size):

			data_,label_,mask_ = self.load_image_label()
			self.data.append(data_)
			self.label.append(label_)
			self.mask_.append(mask_)

		state = np.random.get_state()
		np.random.shuffle(self.data)
		np.random.set_state(state)
		np.random.shuffle(self.label)

		self.data = np.array(self.data)
		self.label = np.array(self.label)
		self.mask_ = np.array(self.mask_)
		# print  'data.shape:', self.data.shape,  'label.shape:', self.label.shape
		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(*self.data.shape)
		top[1].reshape(*self.label.shape)
		top[2].reshape(*self.mask_.shape)


	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label
		top[2].data[...] = self.mask_


	def backward(self, top, propagate_down, bottom):
		pass

	def load_image_label(self):
		"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

		if (self.id +1) >= len(self.train_img_list):
			state = np.random.get_state()
			np.random.shuffle(self.train_img_list)
			np.random.set_state(state)
			np.random.shuffle(self.train_label_list)
			self.id = 0


		#img_l_path = self.train_img_list[self.id%len(self.train_img_list)].strip()
		#label_l_path = self.train_label_list[self.id%len(self.train_img_list)].strip()
		img_r_path = self.train_img_list[self.id].strip()
		label_r_path = self.train_label_list[self.id].strip()
		#print(img_r_path)

		# if os.path.basename(img_l_path) != os.path.basename(label_l_path):
		# 	print('Warning: Img ', os.path.basename(img_l_path), 'Label ', os.path.basename(label_l_path), 'DIFF')
		if os.path.basename(img_r_path.strip('.png')) != os.path.basename(label_r_path.strip('.png')):
			print('Warning: Img ', os.path.basename(img_r_path), 'Label ', os.path.basename(label_r_path), 'DIFF')

		#img_l = cv2.imread(img_l_path)
		img_r = cv2.imread(img_r_path)

		#label_l = Image.open(label_l_path)
		#label_l = np.array(label_l, dtype=np.uint8)
		label_r = Image.open(label_r_path)
		label_r = np.array(label_r, dtype=np.uint8)


		# print 'label_l_path',label_l_path
		# print 'label_r_path',label_r_path
		# print 'label_l.size',label_l.shape
		# print 'label_r.size', label_r.shape

		##filp
		# if np.random.randint(4) == 0:
		# 	img_l = np.flip(img_l,axis=1)
		# 	label_l = np.flip(label_l,axis=1)
		# if np.random.randint(4) == 0:
		# 	img_r = np.flip(img_r,axis=1)
		# 	label_r = np.flip(label_r,axis=1)


		## rotate img_r
		# img_r = np.rot90(img_r, 2)
		# label_r = np.rot90(label_r, 2)

		## merge
		img_all = np.array(img_r).astype(np.uint8)
		#img_all = np.vstack((img_l,img_r)).astype(np.uint8)
		#label_all =np.vstack((label_l,label_r)).astype(np.uint8)
		label_all = label_r

		## rotate img_all
		# img_all = np.rot90(img_all, 2)
		# label_all = np.rot90(label_all, 2)

		# resize
		img_all = cv2.resize(img_all, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
		label_all = Image.fromarray(label_all, 'P')
		label_all = label_all.resize([self.resize_w, self.resize_h], Image.NEAREST)
		label_all = np.array(label_all, dtype=np.uint8)

		## crop
		if self.random_crop:
			img_all,label_all ,mask= self.random_crop_data(img_all,label_all,self.mask)

		# ## test
		# cv2.imshow('img',img_all)
		# cv2.imshow('label',label_all)
		# sys.stdout.flush()
		# sys.stdout.write('shape'+ str(img_all.shape))
		# cv2.waitKey()
		# ##
		self.id +=1
		img_all = img_all.transpose((2, 0, 1))
		label_all = label_all[np.newaxis, ...]
		mask = mask[np.newaxis, ...]


		return img_all,label_all,mask

	def random_crop_data(self,img,label,mask):
		img = np.array(img, dtype=np.uint8)

		max_width = img.shape[1]
		max_height = img.shape[0]

		if ((max_width - self.crop_w) < 0) or ((max_height - self.crop_h)<0):
			print 'random crop size is larger than image size!'

		rand_width = np.random.randint(0, max_width - self.crop_w)
		rand_height = np.random.randint(0, max_height - self.crop_h)

		img = img[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		label = label[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		mask = mask[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]

		return img,label,mask


class BSD_DataLayer_train_mask(caffe.Layer):



	def setup(self, bottom, top):
		"""
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
		''' config '''
		params = eval(self.param_str)
		self.split = params['split']
		self.seed = params.get('seed')
		self.batch_size = params.get('batch_size')
		self.resize_h = params.get('resize_h')
		self.resize_w = params.get('resize_w')
		self.random_crop = params.get('random_crop')
		self.crop_h = params.get('crop_h')
		self.crop_w = params.get('crop_w')
		txt_dir = params.get('txt_dir')
		self.bsd_area = np.asarray([[(150, 20), (46, 450), (960, 460), (800, 66)]], dtype=np.int)
		self.mask = np.zeros((384, 768))
		self.bsd_area = (self.bsd_area * 0.75).astype(np.int)
		print(self.bsd_area)
		for points in self.bsd_area:
			cv2.fillPoly(self.mask, [np.array(points)], 1)

		# three tops: data, semantic, geometric
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		img_ped_list = open(os.path.join(txt_dir, 'train_img_L.txt'), 'r').readlines()
		label_ped_list = open(os.path.join(txt_dir, 'train_label_L.txt'), 'r').readlines()
		img_noped_list = open(os.path.join(txt_dir, 'train_img_R.txt'), 'r').readlines()
		label_noped_list = open(os.path.join(txt_dir, 'train_label_R.txt'), 'r').readlines()

		self.train_img_list = img_ped_list + img_noped_list
		self.train_label_list = label_ped_list + label_noped_list
		print("+++++++++++++train_img_list")
		print(len(self.train_img_list))
		print(len(self.train_label_list))
		np.random.seed(self.seed)
		state = np.random.get_state()
		np.random.shuffle(self.train_img_list)
		np.random.set_state(state)
		np.random.shuffle(self.train_label_list)
		self.id = 0

	def reshape(self, bottom, top):
		# load image + label image pair
		self.data = []
		self.label = []
		for i in range(self.batch_size):
			data_, label_ = self.load_image_label()
			self.data.append(data_)
			self.label.append(label_)

		state = np.random.get_state()
		np.random.shuffle(self.data)
		np.random.set_state(state)
		np.random.shuffle(self.label)

		self.data = np.array(self.data)
		self.label = np.array(self.label)
		# print  'data.shape:', self.data.shape,  'label.shape:', self.label.shape
		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(*self.data.shape)
		top[1].reshape(*self.label.shape)

	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label

	def backward(self, top, propagate_down, bottom):
		pass

	def load_image_label(self):
		"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

		if (self.id + 1) >= len(self.train_img_list):
			state = np.random.get_state()
			np.random.shuffle(self.train_img_list)
			np.random.set_state(state)
			np.random.shuffle(self.train_label_list)
			self.id = 0

		# img_l_path = self.train_img_list[self.id%len(self.train_img_list)].strip()
		# label_l_path = self.train_label_list[self.id%len(self.train_img_list)].strip()
		img_r_path = self.train_img_list[self.id].strip()
		label_r_path = self.train_label_list[self.id].strip()
		# print(img_r_path)

		# if os.path.basename(img_l_path) != os.path.basename(label_l_path):
		# 	print('Warning: Img ', os.path.basename(img_l_path), 'Label ', os.path.basename(label_l_path), 'DIFF')
		if os.path.basename(img_r_path.strip('.png')) != os.path.basename(label_r_path.strip('.png')):
			print('Warning: Img ', os.path.basename(img_r_path), 'Label ', os.path.basename(label_r_path), 'DIFF')

		# img_l = cv2.imread(img_l_path)
		img_r = cv2.imread(img_r_path)

		# label_l = Image.open(label_l_path)
		# label_l = np.array(label_l, dtype=np.uint8)
		label_r = Image.open(label_r_path)
		label_r = np.array(label_r, dtype=np.uint8)

		# print 'label_l_path',label_l_path
		# print 'label_r_path',label_r_path
		# print 'label_l.size',label_l.shape
		# print 'label_r.size', label_r.shape

		##filp
		# if np.random.randint(4) == 0:
		# 	img_l = np.flip(img_l,axis=1)
		# 	label_l = np.flip(label_l,axis=1)
		# if np.random.randint(4) == 0:
		# 	img_r = np.flip(img_r,axis=1)
		# 	label_r = np.flip(label_r,axis=1)

		## rotate img_r
		# img_r = np.rot90(img_r, 2)
		# label_r = np.rot90(label_r, 2)

		## merge
		img_all = np.array(img_r).astype(np.uint8)
		# img_all = np.vstack((img_l,img_r)).astype(np.uint8)
		# label_all =np.vstack((label_l,label_r)).astype(np.uint8)
		label_all = label_r

		## rotate img_all
		# img_all = np.rot90(img_all, 2)
		# label_all = np.rot90(label_all, 2)

		# resize
		img_all = cv2.resize(img_all, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
		label_all = Image.fromarray(label_all, 'P')
		label_all = label_all.resize([self.resize_w, self.resize_h], Image.NEAREST)
		label_all = np.array(label_all, dtype=np.uint8)
		label_all [label_all ==3] = 2
		img_all [self.mask ==0] =0

		label_all[self.mask == 0] = 0

		## crop
		if self.random_crop:
			img_all, label_all = self.random_crop_data(img_all, label_all)

		# ## test
		# cv2.imshow('img',img_all)
		# cv2.imshow('label',label_all)
		# sys.stdout.flush()
		# sys.stdout.write('shape'+ str(img_all.shape))
		# cv2.waitKey()
		# ##
		self.id += 1
		img_all = img_all.transpose((2, 0, 1))
		label_all = label_all[np.newaxis, ...]

		return img_all, label_all

	def random_crop_data(self, img, label):
		img = np.array(img, dtype=np.uint8)

		max_width = img.shape[1]
		max_height = img.shape[0]

		if ((max_width - self.crop_w) < 0) or ((max_height - self.crop_h) < 0):
			print 'random crop size is larger than image size!'

		rand_width = np.random.randint(0, max_width - self.crop_w)
		rand_height = np.random.randint(0, max_height - self.crop_h)

		img = img[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		label = label[rand_height:rand_height + self.crop_h:, rand_width:rand_width + self.crop_w]
		return img, label
class BSDLayer_test_combine(caffe.Layer):
	"""
    Load (input image, label image) pairs from SIFT Flow
    one-at-a-time while reshaping the net to preserve dimensions.

    This data layer has three tops:

    1. the data, pre-processed
    2. the label, regression label

    Use this to feed data to a fully convolutional network.
    """

	def setup(self, bottom, top):
		"""
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
		# config
		params = eval(self.param_str)
		# self.dir_ = params['dir']
		self.split = params['split']
		# self.mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
		# self.random = params.get('randomize', True)
		self.seed = params.get('seed')
		self.batch_size = params.get('batch_size')
		self.resize_h = params.get('resize_h')
		self.resize_w = params.get('resize_w')
		self.random_crop = params.get('random_crop')
		self.crop_h = params.get('crop_h')
		self.crop_w = params.get('crop_w')
		txt_dir = params.get('txt_dir')


		# three tops: data, semantic, geometric
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		# load indices for images and labels

		self.img_l_list = open(os.path.join(txt_dir,'test_img_L.txt'), 'r').readlines()
		self.label_l_list = open(os.path.join(txt_dir,'test_label_L.txt'), 'r').readlines()
		self.img_r_list = open(os.path.join(txt_dir,'test_img_R.txt'), 'r').readlines()
		self.label_r_list = open(os.path.join(txt_dir,'test_label_R.txt'), 'r').readlines()
		self.id_l = 0
		self.id_r = 0



	def reshape(self, bottom, top):
		# load image + label image pair
		self.data = []
		self.label = []
		for i in range(self.batch_size):
			self.id_l +=1
			self.id_r +=1
			if self.id_l >= len(self.img_r_list):
				self.id_l =0
				self.id_r =0

			data_,label_ = self.load_image_label()
			self.data.append(data_)
			self.label.append(label_)
		self.data = np.array(self.data)
		self.label = np.array(self.label)
		# print  'data.shape:', self.data.shape,  'label.shape:', self.label.shape
		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(*self.data.shape)
		top[1].reshape(*self.label.shape)

	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label

		# pick next input
		# self.id_l += self.batch_size
		#
		# if self.id_l == len(self.img_list):
		# 	self.id_l = 0
		# if self.id_l + self.batch_size > len(self.img_l_list):
		# 	self.id_l = self.id_l + self.batch_size - len(self.img_l_list)

		# if self.random:
		# 	self.idx = random.randint(0, len(self.indices) - 1)
		# else:
		# 	self.idx += self.batch_size
		# 	if self.idx == len(self.img_list):
		# 		self.idx = 0
		# 	if self.idx + self.batch_size > len(self.img_list):
		# 		self.idx = self.idx + self.batch_size - len(self.img_list)

	def backward(self, top, propagate_down, bottom):
		pass

	def load_image_label(self):
		"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
		# self.id_l = np.random.randint(0, len(self.img_l_list) - 1)
		# self.id_r = np.random.randint(0, len(self.img_r_list) - 1)

		#img_l_path = self.img_l_list[self.id_l].strip()
		#label_l_path = self.label_l_list[self.id_l].strip()
		img_r_path = self.img_r_list[self.id_r].strip()
		label_r_path = self.label_r_list[self.id_r].strip()

		#if os.path.basename(img_l_path) != os.path.basename(label_l_path):
		#	print('Warning: Img ',os.path.basename(img_l_path),'Label ',os.path.basename(label_l_path),'DIFF')
		if os.path.basename(img_r_path) != os.path.basename(label_r_path):
			print('Warning: Img ',os.path.basename(img_r_path),'Label ',os.path.basename(label_r_path),'DIFF')

		#img_l = cv2.imread(img_l_path)
		img_r = cv2.imread(img_r_path)
		#label_l = Image.open(label_l_path)
		#label_l = np.array(label_l, dtype=np.uint8)
		label_r = Image.open(label_r_path)
		label_r = np.array(label_r, dtype=np.uint8)


		# label_path = self.label_list[id_l].strip()
		# label = self.depth_read(label_path)
		# img_path = self.img_list[idx].strip()
		# img = cv2.imread(img_path, 1)
		# # img = img.astype(np.float32)
		# img_size_W, img_size_H = img.shape[1], img.shape[0]

		## rotate img_r
		#img_r = np.rot90(img_r, 2)
		#label_r = np.rot90(label_r,2)


		## merge
		img_all = np.array(img_r).astype(np.uint8)
		label_all =label_r

		## rotate img_all
		#img_all = np.rot90(img_all, 2)
		#label_all = np.rot90(label_all, 2)

		# resize
		img_all = cv2.resize(img_all, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
		label_all = Image.fromarray(label_all, 'P')
		label_all = label_all.resize([self.resize_w, self.resize_h], Image.NEAREST)
		label_all = np.array(label_all, dtype=np.uint8)
		#
		# # ## test
		# # if 'train' in self.split:
		# cv2.imshow('img',img_all)
		# cv2.imshow('label',label_all)
		# sys.stdout.flush()
		# sys.stdout.write('shape' + str(img_all.shape))
		# cv2.waitKey()
		# ##
		img_all = img_all.transpose((2, 0, 1))
		label_all = label_all[np.newaxis, ...]

		return img_all,label_all

