#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: compute_lingshiceshi_FP.py
@time: 5/23/20 10:46 AM
@purpose:
'''
#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: compute_pred_probability_only_right_from_dir.py
@time: 5/13/20 6:28 PM
@purpose:从文件夹中直接读取图片，进行测试，然后从中统计每一个类别的预测值
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
sys.path.insert(0, CAFFE_HOME + 'python')
import caffe





def drawBoundingBox(img, objects, color):
	for obj in objects:
		cv2.rectangle(img,(obj[0],obj[1]),(obj[0]+obj[2],obj[1]+obj[3]),color,2)
	return img

def findObjects(seg,bsd_area,res_width):
	warning_dic = {'right':{2:0,4:0}}
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
					warning_dic['right'][c] += 1
					break
	return objects,warning_dic

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


def update_report_dic(report_dic,warning_tda,warning_gt):
	for side in ['right']:
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
			if (TP+TN+FP+FN!=0 and TP+FP!=0 and TP+FN!=0 ):
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
		if (TP + TN + FP + FN != 0 and TP + FP != 0 and TP + FN != 0):
			ACC = round((TP + TN) / float(TP + TN + FP + FN), 2)
			PRE = round((TP) / float(TP + FP), 2)
			RECALL = round((TP) / float(TP + FN), 2)
			f.write('%s-sum,%s,%s,%s,%s,%s,%s,%s\n' % (cls_dic[cls_id], TP, TN, FP, FN, ACC, PRE, RECALL))
	f.close()

def resize_bsd_area(height,width):
	# bsd_area = np.asarray([[(230, 42), (66, 180), (868, 218), (764, 96)], [(234, 420), (32, 262), (918, 296), (758, 420)]],dtype=np.int)
	bsd_area = np.asarray([[(230, 42), (66, 372), (868, 410), (764, 96)]],dtype=np.int)

	bsd_area = (bsd_area*(width/1024.0)).astype(np.int)

	return bsd_area

def load_net(model,weights):
	os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
	caffe.set_mode_gpu()
	caffe.set_device(1)
	# caffe.set_mode_cpu()
	net = caffe.Net(model, weights, caffe.TEST)
	return net

def predic_img(net,img,res_height,res_width,probability_set):
	image = copy.deepcopy(img)

	image = cv2.resize(image, (res_width, res_height), interpolation=cv2.INTER_LINEAR)

	input_blob = image.transpose((2, 0, 1))
	input_blob = input_blob[np.newaxis, ...]

	blobs = None  # ['prob', 'argMaxOut']
	out = net.forward_all(blobs=blobs, **{net.inputs[0]: input_blob})
	filters = net.blobs['out_deconv_final_up8'].data[0]
	prob = np.subtract(filters, np.max(filters,axis = 0)[np.newaxis,: , ...])
	prob = np.exp(prob)
	softmax_value = np.divide(prob, np.sum(prob,axis= 0)[np.newaxis,:, ...])
	pred_ped = softmax_value[2,:,:]


	if 'argMaxOut' in out:
		prob = out['argMaxOut'][0]
		prediction = prob[0].astype(int)
	else:
		prob = out['prob'][0]
		prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)

	mask = (prediction == 2) & (pred_ped < probability_set)
	prediction[mask] = 0

	return prediction,pred_ped

def res_from_pc(probability_set):
	'''set param'''
	res_height1 = 384
	res_height2 = 192
	res_width = 768
	show_img = False
	save_res = True
	save_dir = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/thresold'
	model = '/media/soterea/Data_ssd/work/YYZhang/train_file/2020/5/train_fusion_higher_feature/sparse/deploy.prototxt'
	weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020/5/sparse/fusion_higher_feature/BSD_train_sparse_d-n_iter_60000.caffemodel'
	net = load_net(model,weights)
	img_dir = '/media/soterea/Data_ssd/work/YYZhang/data/test_data/testfrom_Litao/2020-05-09-09-50-00/right'
	save_dir_img = os.path.join(save_dir,'add_threhold_'+str(probability_set))
	if not os.path.exists(save_dir_img):
		os.makedirs(save_dir_img)
	txt_path = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/thresold/probability.txt'

	#txt_path = save_dir_img + 'probability.txt'

	count = 0
	bsd_area = resize_bsd_area(res_height1, res_width)
	all_probability = []
	for name in os.listdir(img_dir):



		img_path = os.path.join(img_dir,name)
		img_r = cv2.imread(img_path)
		img_all = np.array(img_r).astype(np.uint8)
		img_all= cv2.resize(img_all, (res_width, res_height1), interpolation=cv2.INTER_LINEAR)

		'''get pc res'''
		prediction ,pred_ped= predic_img(net,img_all,res_height1,res_width,probability_set)
		img_pc, warning_pc = draw_res_gt(img_all,prediction,bsd_area,res_width)
		if warning_pc['right'][2] == 0:
			continue
		'''show'''
		count +=1
		#average_pred_probability = pred_ped_sum/person_#概率平均
		# f = open(txt_path, 'a+')
		#
		# f.write(str(warning_pc['right'][2]))
		# f.write('\n')
		# f.close()
		plt.figure()
		plt.title('Blend_Pre')
		plt.imshow(cv2.cvtColor(img_pc, cv2.COLOR_BGR2RGB))

		plt.tight_layout()
		if show_img:
			plt.show()
		elif save_res:
			plt.savefig(os.path.join(save_dir_img, name))
		plt.close()
	f = open(txt_path, 'a+')

	f.write(str(count)+','+str(probability_set))
	f.write('\n')
	f.close()

if __name__ == '__main__':
	# res_from_pc()
	import numpy as np

	for probability_set in np.arange(0.6,0.85,0.01):

		res_from_pc(round(probability_set,2))

