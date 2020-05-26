#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: test_cv_read_avi2img.py
@time: 3/4/20 12:06 PM
@desc:
'''

import cv2
import string, random

import os
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))


cap_ritht = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/2_right.avi")
cap_left = cv2.VideoCapture("/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/2_left.avi")
count = 0
while (cap_ritht.isOpened()and cap_left.isOpened()):
	ret_right, frame_right = cap_ritht.read()
	ret_left, frame_left = cap_left.read()
	count = count+1
	print (count)
	frame_right = cv2.resize(frame_right,(768,192,),interpolation=cv2.INTER_LINEAR)
	frame_left = cv2.resize(frame_left,(768,192,),interpolation=cv2.INTER_LINEAR)

	cv2.imwrite(os.path.join('/media/soterea/Data_ssd/work/YYZhang/data/test_data/read_video','right_'+str(count)+'.jpg') , frame_right)
	cv2.imwrite(os.path.join('/media/soterea/Data_ssd/work/YYZhang/data/test_data/read_video','left_'+str(count)+'.jpg') , frame_left)

# cap.release()
# cv2.destroyAllWindows()
