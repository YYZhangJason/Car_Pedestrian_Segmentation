#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: test_model_on_YUV_img.py
@time: 3/4/20 12:02 PM
@desc:
'''
#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: test_newmodel.py
@time: 2/15/20 5:44 PM
@desc:
'''
# -*- coding:utf-8 -*-
# Analyze data
# Version: 1.0
# Artur Filipowicz

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
# import caffe.proto.caffe_pb2 as caffe_pb2


def pixelTo3D(p,H,T):
	P = np.dot(H,[p[0],p[1],1])
	P = P*T[2]/P[2]+T
	return P

def convertObjs(objs,H,T):
	cObjs = list()
	for obj in objs:
		if (obj[1] > 256):
			c = 512-obj[1]
			p = pixelTo3D([obj[0],c],H,T)
			p_1 = pixelTo3D([obj[0]+obj[2],c],H,T)
			p[1] = -p[1]
			cObjs.append((p[0]+4.13,p[1],p_1[0]-p[0],obj[4]))
		else:
			p = pixelTo3D([obj[0],obj[1]+obj[3]],H,T)
			p_1 = pixelTo3D([obj[0]+obj[2],obj[1]+obj[3]],H,T)
			cObjs.append((p[0]+4.13,p[1],p_1[0]-p[0],obj[4]))
	return cObjs

def BSDAlarm(objs):
	for obj in objs:
		if (np.abs(obj[1]) <= 3 and 8.72+3 >= obj[0] and obj[0] >= -3 ):
			return 1
	return 0




def drawBoundingBox(img, objects, color):
	for obj in objects:
		cv2.rectangle(img,(obj[0],obj[1]),(obj[0]+obj[2],obj[1]+obj[3]),color,2)
	return img

def findObjects(seg,bsd_area,res_width):
	warning_dic = {'left':{2:0,4:0},'right':{2:0,4:0}}
	mask = np.zeros(seg.shape, np.uint8)
	for pts in bsd_area:
		cv2.fillPoly(mask,[np.array(pts)],1)

	objects = {2:[],4:[]}
	for c in objects:
		classSeg = np.zeros(seg.shape, np.uint8)
		classSeg[seg==c] = 255
		contours,hierarchy = cv2.findContours(classSeg, 1, 1)
		for cnt in contours:
			if cv2.contourArea(cnt) < 500*(res_width/1024.0):
				continue
			for pt in cnt:
				if mask[pt[0][1],pt[0][0]] == 1:
					x,y,w,h = cv2.boundingRect(cnt)
					objects[c].append((x,y,w,h,c))
					if y >seg.shape[0]//2:
						warning_dic['left'][c] =1
					else:
						warning_dic['right'][c] = 1
					break
	return objects,warning_dic


def get_TDA_array(path,height,width,):
	fp = open(path, 'rb')
	frame_len = height * width  # 一帧图像所含的像素个数

	fp.seek(0, 2)  # 设置文件指针到文件流的尾部
	ps = fp.tell()  # 当前文件指针位置
	numfrm = ps // frame_len  # 计算输出帧数
	fp.seek(0, 0)

	return fp,numfrm

def get_record_dic(path):
	'''
	id_num:,img_l_path,img_r_path,label_l_path,label_r_path

	'''
	f_dic = {}
	f = open(path,'r').readlines()
	for i in range(1,len(f)):
		line = f[i].strip()
		img_id = int(line.split(',')[0])
		f_dic[img_id] = line.split(',')[1:]

	return f_dic


def decode_tda_info(tda_preInfo):
	res_dic = {'left':[],'right':[]}
	warning_dic = {'left':{2:0,4:0},'right':{2:0,4:0}}

	for side in ['left','right']:
		if side == 'right':
			tda_preInfo_side = tda_preInfo[1024*2:]
		else:
			tda_preInfo_side = tda_preInfo

		[det_num] = np.frombuffer(tda_preInfo_side[:4], np.uint32)
		for i in range(det_num):
			[x, y, l] = np.frombuffer(tda_preInfo_side[4 + i * 16:4 + i * 16 + 12], np.float32)
			[c] = np.frombuffer(tda_preInfo_side[16 + i * 16:16 + i * 16 + 4], np.uint32)

			warning_dic[side][c] = 1

			[x1, y1, x2, y2] = np.frombuffer(tda_preInfo_side[1024 + i * 16:1024 + i * 16 + 16], np.uint32)

			res_dic[side].append([x, y, l, c, x1, y1, x2, y2])

	return res_dic,warning_dic


def get_recordImg(record_dic,frm_id,resize_width,resize_height):
	line = record_dic[frm_id]
	img_l,img_r,label_l,label_r = line[0],line[1],line[2],line[3]

	if 'bsd_night' in img_l:
		time_condi = 'night'
	elif 'bsd_day' in img_l:
		time_condi = 'day'
	else:
		print('warning,can not know time condition:',img_l)


	img_l = cv2.imread(img_l)
	label_l = Image.open(label_l)
	img_r = cv2.imread(img_r)
	label_r = Image.open(label_r)

	img_l = np.rot90(img_l, 2)
	label_l = np.rot90(label_l, 2)

	img_all = np.vstack((img_r, img_l)).astype(np.uint8)
	label_all = np.vstack((label_r, label_l)).astype(np.uint8)

	original_size = img_all.shape
	img_all_resize = cv2.resize(img_all, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
	label_all = Image.fromarray(label_all, 'P')
	label_all = label_all.resize([resize_width, resize_height], Image.NEAREST)
	label_all = np.array(label_all, dtype=np.uint8)

	#return img_all_resize,label_all,time_condi
	return label_all


def chroma_blend(image, img_id):
	# 0:empty,1:road,2:ped,3:unused,4:car
	palette_ = [[0, 0, 0], [160, 32, 240], [255, 0, 0], [255, 0, 0], [0, 0, 255], [255, 165, 0]]
	palette = np.zeros((256, 3),np.uint)
	for i, p in enumerate(palette_):
		palette[i, 0] = p[0]
		palette[i, 1] = p[1]
		palette[i, 2] = p[2]
	palette = palette[..., ::-1]

	color = palette[img_id.ravel()].reshape(image.shape)

	image_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV)
	image_y,image_u,image_v = cv2.split(image_yuv)
	color_yuv = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_BGR2YUV)
	color_y,color_u,color_v = cv2.split(color_yuv)
	image_y = np.uint8(image_y)
	color_u = np.uint8(color_u)
	color_v = np.uint8(color_v)
	image_yuv = cv2.merge((image_y,color_u,color_v))
	image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
	return image


def draw_res_gt(image,label,bsd_area,res_width):
	color_dic = {2:[0,0,255],4:[255,0,0]}

	img = copy.deepcopy(image)

	dets,warning_dic_gt = findObjects(label,bsd_area,res_width)

	for pts in bsd_area:
		pts = np.array(pts, np.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(img, [pts], True, (0, 255, 0), 2)

	for cls in dets:
		drawBoundingBox(img, dets[cls], color_dic[cls])

	return img,warning_dic_gt



def draw_res_tda(image,tda_preInfo,bsd_area):
	'''
	ped = [0,0,255]
	car = [0,255,0]
	'''
	color_dic = {2:[0,0,255],4:[255,0,0]}
	img = copy.deepcopy(image)

	for pts in bsd_area:
		pts = np.array(pts, np.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(img, [pts], True, (0, 255, 0), 2)

	for side in tda_preInfo:
		for det in tda_preInfo[side]:
			c,x1,y1,x2,y2 = det[3],det[4],det[5],det[6],det[7]
			# cv2.rectangle(img,(x1,y1),[x2,y2],color_dic[c],2)
			cv2.rectangle(img,(x1,y1),(x2,y2),color_dic[c],2)

	return img


def update_report_dic(report_dic,warning_tda,warning_gt):
	for side in ['left','right']:
		for cls_id in [2,4]:
		#for cls_id in [2]:
			print('+++++++++++++++++++++++++++')
			print(warning_gt[side][cls_id] )
			print(warning_tda[side][cls_id])
			if warning_gt[side][cls_id] == warning_tda[side][cls_id]:
				if warning_gt[side][cls_id] == 1:
					report_dic[side][cls_id]['TP']+=1
				else:report_dic[side][cls_id]['TN']+=1
			else:
				if warning_gt[side][cls_id] == 1:
					report_dic[side][cls_id]['FN']+=1
				else:report_dic[side][cls_id]['FP']+=1

	return report_dic


def report_dic2csv(report_dic,save_dir,time_condi):
	save_dir = os.path.join(save_dir,'report')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	cls_dic = {2:'Ped',4:'Car'}
	#cls_dic = {2: 'Ped'}
	f = open(os.path.join(save_dir,'report_%s.csv'%(time_condi)),'w')
	f.write(',TP,TN,FP,FN,ACC,PRE,RECALL\n')
	sum_dic = {2:{},4:{}}
	#sum_dic = {2: {}}
	for side in report_dic:
		for cls_id in report_dic[side]:
			TP,TN,FP,FN = report_dic[side][cls_id]['TP'],\
						  report_dic[side][cls_id]['TN'],\
						  report_dic[side][cls_id]['FP'],\
						  report_dic[side][cls_id]['FN']

			ACC = round((TP+TN)/float(TP+TN+FP+FN),2)
			PRE = round((TP)/float(TP+FP),2)
			RECALL = round((TP)/float(TP+FN),2)
			f.write('%s-%s,%s,%s,%s,%s,%s,%s,%s\n'%(cls_dic[cls_id],side,TP,TN,FP,FN,ACC,PRE,RECALL))

			if 'TP' in sum_dic[cls_id]:
				sum_dic[cls_id]['TP'] += TP
				sum_dic[cls_id]['TN'] += TN
				sum_dic[cls_id]['FP'] += FP
				sum_dic[cls_id]['FN'] += FN
			else:
				sum_dic[cls_id]['TP'] = TP
				sum_dic[cls_id]['TN'] = TN
				sum_dic[cls_id]['FP'] = FP
				sum_dic[cls_id]['FN'] = FN

	for cls_id in cls_dic:
		TP, TN, FP, FN = sum_dic[cls_id]['TP'], \
						 sum_dic[cls_id]['TN'], \
						 sum_dic[cls_id]['FP'], \
						 sum_dic[cls_id]['FN']
		ACC = round((TP + TN) / float(TP + TN + FP + FN), 2)
		PRE = round((TP) / float(TP + FP), 2)
		RECALL = round((TP) / float(TP + FN), 2)
		f.write('%s-sum,%s,%s,%s,%s,%s,%s,%s\n' % (cls_dic[cls_id], TP, TN, FP, FN, ACC, PRE, RECALL))
	f.close()

def resize_bsd_area(height,width):
	bsd_area = np.asarray([[(230, 42), (66, 180), (868, 218), (764, 96)], [(234, 420), (32, 262), (918, 296), (758, 420)]],dtype=np.int)
	bsd_area = (bsd_area*(width/1024.0)).astype(np.int)

	return bsd_area

def load_net(model,weights):
	os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
	caffe.set_mode_gpu()
	caffe.set_device(0)

	net = caffe.Net(model, weights, caffe.TEST)

	return net

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

def CLAHE (img):
	b,g,r = cv2.split(img)
	clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	b = clahe.apply(b)
	g = clahe.apply(g)
	r = clahe.apply(r)
	image = cv2.merge([b,g,r])

	return image
def contact_left_right(img_l,img_r,resize_width,resize_height):
	img_l = np.rot90(img_l, 2)

	img_all = np.vstack((img_r, img_l)).astype(np.uint8)

	original_size = img_all.shape
	img_all_resize = cv2.resize(img_all, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)


	#return img_all_resize,label_all,time_condi
	return img_all_resize
def res_from_pc():
	'''set param'''
	res_height1 = 384
	res_height2 = 192
	res_width = 768
	show_img = False
	save_res = True
	model = '/media/soterea/Data_ssd/work/YYZhang/train_file/train_data_augmentation/sparse/deploy.prototxt'
	weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel'
	net = load_net(model,weights)
	bsd_area = resize_bsd_area(res_height1, res_width)
	cap_ritht = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/4/10_right.avi")
	cap_left = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/4/10_left.avi")
	count = 0
	while (cap_ritht.isOpened() and cap_left.isOpened()):
		count += 1
		ret_right, frame_right = cap_ritht.read()
		ret_left, frame_left = cap_left.read()
		frame = contact_left_right(frame_left, frame_right, res_width, res_height1)
		'''get pc res'''
		prediction = predic_img(net,frame,res_height1,res_width)

		img_pc, warning_pc = draw_res_gt(frame,prediction,bsd_area,res_width)
		blend_pre = chroma_blend(frame, prediction)
		cv2.imwrite(os.path.join('/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/yuv_test_result_10',str(count)+'.png'),img_pc)

def main():
	'''set param'''
	res_height1 = 384
	res_height2 = 192
	res_width = 768
	show_img = False
	save_res = True
	model = '/media/soterea/Data_ssd/work/YYZhang/train_file/train_data_augmentation/sparse/deploy.prototxt'
	weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel'
	img_path = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/yuv_no_crotect/left-png'
	save_path = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/yuv_no_crotect/dete_result'
	net = load_net(model, weights)
	bsd_area = resize_bsd_area(res_height1, res_width)
	#cap_ritht = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/4/10_right.avi")
	#cap_left = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/4/10_left.avi")
	#count = 0
	img_list = os.listdir(img_path)
	#while (cap_ritht.isOpened() and cap_left.isOpened()):
	count = 0

	for img_name in img_list:
		count += 1

		frame = np.array(cv2.imread(os.path.join(img_path,img_name))).astype(np.uint8)
		frame = cv2.resize(frame, (res_width, res_height1), interpolation=cv2.INTER_LINEAR)
		'''get pc res'''
		prediction = predic_img(net, frame, res_height1, res_width)
		img_pc, warning_pc = draw_res_gt(frame, prediction, bsd_area, res_width)
		blend_pre = chroma_blend(frame, prediction)
		cv2.imwrite(os.path.join(save_path,str(count) + '.png'), img_pc)

if __name__ == '__main__':
	#res_from_pc()
	main()
