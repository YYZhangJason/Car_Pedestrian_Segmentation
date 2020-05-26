
#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Eric
@file: test.py
@time: 2/25/20 12:41 PM
@desc:
'''
import cv2
import os


im_dir = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/yuv_test_result_10'

#video_dir = '/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/yuv_test_result_1/highspeed2.avi'

import cv2
image=cv2.imread('/media/soterea/Data_ssd/work/YYZhang/data/YUV_raw_img/YUV/video_data/yuv_test_result_10/1.png')
img_list = os.listdir(im_dir)
length_ = len(img_list)
print length_
#cv2.imshow("new window", image)
image_info=image.shape
height=image_info[0]
width=image_info[1]
size=(height,width)
print(size)
fps = 2
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(im_dir+'/'+'highspeed10.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))


for i in range(1,length_):
    print i
    file_name = im_dir+'/'+ str(i) + '.png'
    image=cv2.imread(file_name)
    video.write(image)
cv2.waitKey()
