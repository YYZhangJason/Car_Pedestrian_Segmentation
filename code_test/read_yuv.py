#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: read_yuv.py
@time: 3/4/20 10:52 AM
@desc:
'''
import cv2
import numpy as np
import os

# read_file = np.fromfile('/media/soterea/Data_ssd/work/YYZhang/test_file/sendtoTDA_data/3.3_img2TDA/test_BGR_20200303.yuv',dtype= np.uint8)#读取yuv文件信息
# print(len(read_file))
"""
A collection of utility functions to deal with the data collect from the labellers.
"""

import cv2
import os
import numpy as np
import shutil
from PIL import Image
import time
import threading


def save_img(img_path, img):
	cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 5])


def undistort(img):
	""" Return the undistorted image """

	# Distortion information
	K = np.array(
		[[534.0521945240896, 0.0, 666.4726464105142], [0.0, 535.114492525197, 369.8939959094109], [0.0, 0.0, 1.0]])
	D = np.array([[-0.055108], [0.0184714], [-0.0208211], [0.0065596]])
	Knew = K.copy()
	Knew[(0, 1), (0, 1)] = 0.4 * Knew[(0, 1), (0, 1)]

	img = cv2.resize(img, (1280, 720))  # resize to reference resolution

	h, w = img.shape[:2]
	img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
	roi = img_undistorted[200:520, 0:1280]

	return roi


def yuv_nv12_2bgr(file_dir, height, width, save_dir, startfrm=0):
	"""
	:param filename: 待处理 YUV 视频的名字
	:param height: YUV 视频中图像的高
	:param width: YUV 视频中图像的宽
	:param startfrm: 起始帧
	:return: None
	"""
	files = os.listdir(file_dir)
	for file in files:
		if '.yuv' not in file:
			continue
		fp = open(os.path.join(file_dir, file), 'rb')

		# Define the codec and create VideoWriter object
		# fourcc = cv2.cv.CV_FOURCC(*'DIVX') # python 2 opencv
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Python 3 opencv

		fps = 10
		leftVideoName = os.path.join(file_dir, file.replace(".yuv", "_left.avi"))
		rightVideoName = os.path.join(file_dir, file.replace(".yuv", "_right.avi"))
		videoOutLeft = cv2.VideoWriter(leftVideoName, fourcc, float(fps),(1280, 320))
		videoOutRight = cv2.VideoWriter(rightVideoName, fourcc, float(fps), (1280, 320))
		frame_num = 0

		frame_len = height * width * 3 // 2  # 一帧图像所含的像素个数
		shape = (int(height * 1.5), width)

		fp.seek(0, 2)  # 设置文件指针到文件流的尾部
		ps = fp.tell()  # 当前文件指针位置
		numfrm = ps // frame_len  # 计算输出帧数
		fp.seek(frame_len * startfrm, 0)

		t_start = time.time()

		thre = []
		for i in range(numfrm - startfrm):

			raw = fp.read(frame_len)
			yuv = np.frombuffer(raw, np.uint8)
			yuv = yuv.reshape(shape)

			if yuv[0, 0] == 255:
				img_side = 'left'
			elif yuv[0, 0] == 0:
				img_side = 'right'
			# print(yuv[0,0])

			bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # 注意 YUV 的存储格式
			undistorted_img = undistort(bgr_img)

			if img_side == 'left':
				videoOutLeft.write(undistorted_img)
				frame_num += 1
				if frame_num % 100 == 0:
					print("Processing Frame:", frame_num, "of", file)
			else:
				videoOutRight.write(undistorted_img)

		videoOutLeft.release()
		videoOutRight.release()


if __name__ == '__main__':
	input_dir = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/yuv_no_crotect/left-yuv/'
	save_dir = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/yuv_no_crotect/left-yuv/sparse_yuv'
	yuv_nv12_2bgr(file_dir=input_dir, height=720, width=1280, save_dir=save_dir, startfrm=0)

