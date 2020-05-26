#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: img2TDA.py
@time: 3/3/20 4:00 PM
@desc:
'''
#coding=utf-8
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
import time


# @profile #kernprof -l -v
def convert_img2yuv(img_dir,save_dir):

	img_list = sorted(os.listdir(img_dir))
	img_list = img_list[:500]
	output_yuv = np.array([])


	t_start = time.time()

	counter = 0
	for file in img_list:
		counter +=1
		img_path = os.path.join(img_dir,file)
		img_raw = cv2.imread(img_path)
		img_yuv_yv12 = cv2.cvtColor(img_raw,cv2.COLOR_BGR2YUV_YV12)
		img_yuv_nv12 = convert_yv12_to_nv12(img_yuv_yv12)
		img_out = img_yuv_nv12

		if output_yuv.size ==0:
			output_yuv = img_out
		else:
			output_yuv = np.vstack((output_yuv,img_out))

		if counter % 50 ==0:
			print('writing: %d images'%(counter))

	t_end = time.time()
	print('runing time:%f.3 ' % (t_end - t_start))

	output_yuv = np.asarray(output_yuv,dtype=np.uint8)
	output_yuv.astype(np.uint8).tofile('test3.yuv')


def convert_img2yuv_new(img_dir,save_dir):
	img_list = sorted(os.listdir(img_dir))

	## for test
	img_list = img_list[:500]

	## get image.shape
	img_temp = cv2.imread(os.path.join(img_dir,img_list[0]))
	img_h,img_w = img_temp.shape[0],img_temp.shape[1]

	output_yuv = np.zeros((len(img_list)*img_h*3//2,img_w),np.uint8)

	t_start = time.time()
	counter = 0
	for file in img_list:

		img_path = os.path.join(img_dir,file)
		img_raw = cv2.imread(img_path)
		img_yuv_yv12 = cv2.cvtColor(img_raw,cv2.COLOR_BGR2YUV_YV12)
		img_yuv_nv12 = convert_yv12_to_nv12(img_yuv_yv12)
		img_out = img_yuv_nv12

		output_yuv[counter*img_h*3//2:(counter+1)*img_h*3//2,:] = img_out

		counter +=1
		if counter % 50 ==0:
			print('writing: %d images'%(counter))

	t_end = time.time()
	print('runing time:%f.3 ' % (t_end - t_start))

	output_yuv = np.asarray(output_yuv,dtype=np.uint8)
	output_yuv.astype(np.uint8).tofile('test3.yuv')



def convert_yv12_to_nv12(img_yuv_yv12):
	'''
	convert YUV_YV12 FORMAT INTO YUV_NV12 FORMAT
	:param img_yuv_yv12:
	:return:
	'''
	h,w = img_yuv_yv12.shape[0],img_yuv_yv12.shape[1]
	img_y = img_yuv_yv12[:int(h*2/3),:]
	img_u =  img_yuv_yv12[int(h*2/3):int(h*5/6),:]
	img_v = img_yuv_yv12[int(h*5/6):,:]
	img_u = img_u.reshape((img_u.shape[0]*2,int(img_u.shape[1]/2)))
	img_v = img_v.reshape((img_v.shape[0]*2,int(img_v.shape[1]/2)))

	## become UVUVUV....
	img_uv = np.insert(img_u,range(int(w/2)),img_v,axis=1)
	img_yuv_nv12 = np.vstack((img_y,img_uv))

	return img_yuv_nv12


def save_img(img_path,img):
	cv2.imwrite(img_path,img)


def yuv_nv12_2bgr(file_dir, height, width,save_dir, startfrm=0):
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
		fp = open(os.path.join(file_dir,file), 'rb')

		frame_len = height * width * 3 // 2 # 一帧图像所含的像素个数
		shape = (int(height * 1.5), width)

		fp.seek(0, 2)  # 设置文件指针到文件流的尾部
		ps = fp.tell()  # 当前文件指针位置
		numfrm = ps // frame_len  # 计算输出帧数
		fp.seek(frame_len * startfrm, 0)

		t_start = time.time()

		thre = []
		for i in range(numfrm - startfrm):

			raw = fp.read(frame_len)
			yuv = np.frombuffer(raw,np.uint8)
			yuv = yuv.reshape(shape)

			bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # 注意 YUV 的存储格式
			bgr_img_path = os.path.join(save_dir,file,'%s_%d.png' % (file,i + 1))
			if not os.path.exists(os.path.join(save_dir,file)):
				os.makedirs(os.path.join(save_dir,file))

			process = threading.Thread(target=save_img,args=(bgr_img_path,bgr_img))
			thre.append(process)
			process.start()
			# save_img(bgr_img_path,bgr_img)
			# cv2.imwrite(bgr_img_path, bgr_img)
			print("Extract frame %d " % (i + 1))

		for process in thre:
			process.join()

		t_end = time.time()
		print('runing time:%f.3 for %s'%(t_end-t_start,file))

		fp.close()
	print("job done!")
	return None

def main(resize_width,resize_height,img_l_txt,img_r_txt,label_l_txt,label_r_txt,save_dir,num=None):
	save_dir_label = os.path.join(save_dir,'label')
	save_dir_img = os.path.join(save_dir,'raw')
	if not os.path.exists(save_dir_img):
		os.makedirs(save_dir_img)
		os.makedirs(save_dir_label)

	img_l_list = open(img_l_txt, 'r').readlines()
	img_r_list = open(img_r_txt, 'r').readlines()
	label_l_list = open(label_l_txt, 'r').readlines()
	label_r_list = open(label_r_txt, 'r').readlines()

	## check l nums == r nums
	if not (len(img_l_list) == len(img_r_list) == len(label_l_list) == len(label_r_list)):
		print('number of left & right is not equal!')
		return

	## create big file for TDA RGB map
	if num:
		TDA_RGB_map = np.zeros((resize_height*3*num,resize_width),np.uint8)
	else:
		TDA_RGB_map = np.zeros((resize_height*3*len(img_l_list),resize_width),np.uint8)

	f_record = open(os.path.join(save_dir,'record.csv'),'w')
	f_record.write('id_num:,img_l_path,img_r_path,label_l_path,label_r_path\n')

	count = 0
	for id in range(len(img_l_list)):
		count += 1
		if num and count>num:
			break
		t_start = time.time()

		img_l_path = img_l_list[id].strip()
		img_r_path = img_r_list[id].strip()
		label_l_path = label_l_list[id].strip()
		label_r_path = label_r_list[id].strip()

		img_l = cv2.imread(img_l_path)
		label_l = Image.open(label_l_path)
		img_r = cv2.imread(img_r_path)
		label_r = Image.open(label_r_path)

		## rotate img
		img_l = np.rot90(img_l, 2)
		label_l = np.rot90(label_l, 2)

		## 2 image
		img_all = np.vstack((img_r, img_l)).astype(np.uint8)
		label_all = np.vstack((label_r, label_l)).astype(np.uint8)

		original_size = img_all.shape
		img_all_resize = cv2.resize(img_all, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
		print("+++++++")
		print(img_all_resize.shape)
		TDA_RGB = covert2TDA_RGB(img_all_resize,mode='BGR')
		# TDA_test = test_convertTDA_RGBback(TDA_RGB)

		# cv2.imshow('RAW',img_all_resize)
		# cv2.imshow('test',TDA_test)
		# cv2.waitKey()

		TDA_RGB_map[id*resize_height*3:(id+1)*resize_height*3,:]=TDA_RGB[:,:]

		cv2.imwrite(os.path.join(save_dir_img,str(id)+'.png'),img_all_resize)
		cv2.imwrite(os.path.join(save_dir_label,str(id)+'.png'),label_all)
		print(save_dir_label,(str(id)+'.png'))

		f_record.write('%s,%s,%s,%s,%s\n'%(id,img_l_path,img_r_path,label_l_path,label_r_path))

	TDA_RGB_map.astype(np.uint8).tofile(os.path.join(save_dir,'test_BGR_%s.yuv')%(time.strftime("%Y%m%d", time.localtime()) ))




def test_convertTDA_RGBback(img):
	img_h,img_w = img.shape[0]//3,img.shape[1]
	out_img = np.zeros((img_h,img_w,3),np.uint8)

	out_img[:, :, 0] = img[:img_h, :]
	out_img[:, :, 1] = img[img_h:img_h * 2, :]
	out_img[:, :, 2] = img[img_h * 2:, :]

	return out_img


def covert2TDA_RGB(img,mode='BGR'):
	'''
	convert 3 channel RGB into stacked TDA RGB map
	:param img:
	:return:
	'''
	img_h,img_w = img.shape[0],img.shape[1]
	out_img = np.zeros((img_h*3,img_w),np.uint8)

	if mode == 'RGB':
		out_img[:img_h,:] = img[:,:,2]
		out_img[img_h:img_h*2,:] = img[:,:,1]
		out_img[img_h*2:,:] = img[:,:,0]

	elif mode == 'BGR':
		out_img[:img_h,:] = img[:,:,0]
		out_img[img_h:img_h*2,:] = img[:,:,1]
		out_img[img_h*2:,:] = img[:,:,2]


	return out_img


if __name__ == '__main__':
	resize_width = 768
	resize_height = 384

	img_dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/train_txt/all_20191217'
	img_l_txt = os.path.join(img_dir,'test_img_L.txt')
	img_r_txt = os.path.join(img_dir,'test_img_R.txt')
	label_l_txt = os.path.join(img_dir,'test_label_L.txt')
	label_r_txt = os.path.join(img_dir,'test_label_R.txt')
	save_dir = '/media/soterea/Data_ssd/work/YYZhang/test_file/sendtoTDA_data/3.3_img2TDA'
	main(resize_width,resize_height,img_l_txt,img_r_txt,label_l_txt,label_r_txt,save_dir)



