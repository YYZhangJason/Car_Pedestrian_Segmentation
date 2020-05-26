#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: deal_with_gao.py
@time: 3/3/20 6:53 PM
@desc:
'''

import cv2
import os
def CLAHE (img):
	b,g,r = cv2.split(img)
	clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	b = clahe.apply(b)
	g = clahe.apply(g)
	r = clahe.apply(r)
	image = cv2.merge([b,g,r])

	return image

def main():
	img_lsit = os.listdir('/media/soterea/Data_ssd/work/YYZhang/test/111')
	for i in range(len(img_lsit)):
		img_name = os.path.join('/media/soterea/Data_ssd/work/YYZhang/test/111',str(i)+'.png')
		image  = cv2.imread(img_name)
		img_deal = CLAHE(image)
		cv2.imwrite(os.path.join('/media/soterea/Data_ssd/work/YYZhang/test/imgdealwithgao',str(i)+'.png'),img_deal)



if __name__ == '__main__':
	main()