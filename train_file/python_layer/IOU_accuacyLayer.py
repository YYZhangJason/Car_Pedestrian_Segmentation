import sys
CAFFE_HOME = '/media/artur/Data_ssd/04_Caffe_Folder/caffe-jacinto/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_HOME + 'python')

import caffe
import numpy as np
from PIL import Image
import os
import cv2


class IOU_accuacyLayer(caffe.Layer):
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

		self.num_labels = 5
		self.ignore_label =  params['ignore_label']
		self.display_iter = params['display_iter']
		self.test_iter_num = params['test_iter_num']

		## class name shouldn't be too long
		self.class_dic = {0:'Empty',1:'road',2:'Ped',3:'unused',4:'Car'}

		self.confusion_matrix_test = np.zeros((self.num_labels, self.num_labels + 1))

		self.test_iter = 0
		self.train_iter = 0

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		# assign output
		if 'train' in self.split:
			self.train_iter +=1
		elif 'test' in self.split:
			self.test_iter +=1

		if (self.train_iter % self.display_iter ==0) and ('train' in self.split):
			pred_batch = bottom[0].data
			label_batch = bottom[1].data
			pred_argmax = np.argmax(pred_batch,axis=1)
			label_batch[label_batch == 3] = 2

			## get confusion_matrix
			confusion_matrix = self.compute_confusion_matrix(label_batch,pred_argmax)
			self.compute_accraucy(confusion_matrix)

		elif 'test' in self.split:
			pred_batch = bottom[0].data
			label_batch = bottom[1].data
			pred_argmax = np.argmax(pred_batch,axis=1)
			label_batch[label_batch == 3] = 2

			# sys.stdout.flush()
			# sys.stdout.write('\n'+str(self.confusion_matrix_test)+'\n')

			confusion_matrix = self.compute_confusion_matrix(label_batch, pred_argmax)
			self.confusion_matrix_test += confusion_matrix

			if self.test_iter % self.test_iter_num == 0:
				confusion_matrix = self.confusion_matrix_test
				self.compute_accraucy(confusion_matrix)
				self.confusion_matrix_test = np.zeros((self.num_labels, self.num_labels + 1))



	def backward(self, top, propagate_down, bottom):
		pass

	def compute_accraucy(self,confusion_matrix):
		## compute iou
		num_classes = self.num_labels
		tp = np.zeros(num_classes)
		population = np.zeros(num_classes)
		det = np.zeros(num_classes)

		for r in range(num_classes):
			for c in range(num_classes):
				population[r] += confusion_matrix[r][c]
				det[c] += confusion_matrix[r][c]
				if r == c:
					tp[r] += confusion_matrix[r][c]

		accuracy_dic = {}
		iou_dic = {}
		for cls in range(num_classes):
			intersection = tp[cls]
			union = population[cls] + det[cls] - tp[cls]
			iou_dic['cls_' + str(cls)] = round((intersection / union), 3) if union else -1
			accuracy_dic['cls_' + str(cls)] = round((intersection / population[cls]), 3) if population[cls] else -1

		sys.stdout.flush()
		if 'train' in self.split:
			sys.stdout.write('\n' + 'TRAIN - current iteration-train:' + str(self.train_iter) + '\n')
		elif 'test' in self.split:
			sys.stdout.write('\n' + 'TEST - current iteration:' + str(self.test_iter) + '\n')
		sys.stdout.write('\t' * 2)
		for i in range(num_classes):
			sys.stdout.write(self.class_dic[i] + '\t')
		sys.stdout.write('\nAccuracy:\t')
		for i in range(num_classes):
			sys.stdout.write(str(accuracy_dic['cls_' + str(i)]) + '\t')
		sys.stdout.write('\nIou:     \t')
		for i in range(num_classes):
			sys.stdout.write(str(iou_dic['cls_' + str(i)]) + '\t')
		sys.stdout.write('\n')

	def compute_confusion_matrix(self,label_batch,pred_argmax):
		confusion_matrix = np.zeros((self.num_labels, self.num_labels + 1))
		for pred, label in zip(pred_argmax, label_batch):
			label = label[0, :, :]
			label = label.ravel()
			pred = pred.ravel().clip(0, self.num_labels)

			label_valid_mask = np.where(label != self.ignore_label)
			labels_valid = label[label_valid_mask]
			pred_valid = pred[label_valid_mask]

			## r=label c=prediction
			for r in range(confusion_matrix.shape[0]):
				for c in range(confusion_matrix.shape[1]):
					confusion_matrix[r, c] += np.sum((labels_valid == r) & (pred_valid == c))

		return confusion_matrix

