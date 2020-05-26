#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: convert_data2train_format.py
@time: 3/25/20 10:05 AM
@purpose:将涛哥传过来的数据进行转换，因为现在标记的时候是标记了8个类别，但是现在我们只使用三个类别进行训练。
'''
"""
A collection of utility functions to deal with the data collect from the labellers.
"""

import cv2
import os
import numpy as np
import shutil
from PIL import Image
import convertColorToID as cvtCol2ID


def move2sameFolder(img_dir,label_dir,save_Dir):
    '''
    Usually first step.
    move all the image,label data into one folder
    :return:
    '''
    save_labelDir = os.path.join(save_Dir,'gt_color')
    save_imgDir = os.path.join(save_Dir,'raw_img')

    if not os.path.exists(os.path.join(save_labelDir,'left')):
        os.makedirs(os.path.join(save_labelDir,'left'))
        os.makedirs(os.path.join(save_labelDir, 'right'))
    if not os.path.exists(os.path.join(save_imgDir,'left')):
        os.makedirs(os.path.join(save_imgDir,'left'))
        os.makedirs(os.path.join(save_imgDir, 'right'))

    ## usually some image not finish label
    f_namelist = []
    f_namelist_img = []

    for subdir, dirs, files in os.walk(label_dir):
        for file in files:
            if 'left.png' in file:
                f_path = os.path.join(subdir,file)
                f_path_save = os.path.join(save_labelDir,'left',file)
                shutil.copy(f_path,f_path_save)
                f_namelist.append(file)
            elif 'right.png' in file:
                f_path = os.path.join(subdir,file)
                f_path_save = os.path.join(save_labelDir,'right',file)
                shutil.copy(f_path,f_path_save)
                f_namelist.append(file)

    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file in f_namelist:
                if 'left.png' in file:
                    f_path = os.path.join(subdir,file)
                    f_path_save = os.path.join(save_imgDir,'left',file)
                    shutil.copy(f_path,f_path_save)
                    f_namelist_img.append(file)
                elif 'right.png' in file:
                    f_path = os.path.join(subdir,file)
                    f_path_save = os.path.join(save_imgDir,'right',file)
                    shutil.copy(f_path,f_path_save)
                    f_namelist_img.append(file)

    for f in f_namelist:
        if f not in f_namelist_img:
            print('no raw-image for: '+ f)
            if 'left' in f:
                f_path_save = os.path.join(save_labelDir, 'left', f)
            else:
                f_path_save = os.path.join(save_labelDir, 'right', f)
            os.remove(f_path_save)


    return save_imgDir,save_labelDir


def make_list(test_per,img_dir,label_dir,save_Dir):
    '''
    given image & test dir, create train & test text file based on test percent
    for caffe python layer as a training path input
    :param test_per:
    :param img_dir:
    :param label_dir:
    :param save_Dir:
    :return:
    '''
    train_img_L = open(os.path.join(save_Dir,'train_img_L.txt'),'w')
    train_label_L = open(os.path.join(save_Dir,'train_label_L.txt'),'w')
    test_img_L = open(os.path.join(save_Dir,'test_img_L.txt'),'w')
    test_label_L = open(os.path.join(save_Dir,'test_label_L.txt'),'w')
    train_img_R = open(os.path.join(save_Dir,'train_img_R.txt'),'w')
    train_label_R = open(os.path.join(save_Dir,'train_label_R.txt'),'w')
    test_img_R = open(os.path.join(save_Dir,'test_img_R.txt'),'w')
    test_label_R = open(os.path.join(save_Dir,'test_label_R.txt'),'w')

    img_list_L = []
    img_list_R = []
    label_list_L = []
    label_list_R = []


    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if not '.png' in file:
                continue

            img_path = os.path.join(subdir, file)
            label_path = os.path.join(label_dir, file)

            if not os.path.exists(label_path):
                print('!warmming:', label_path)
            else:
                if 'left' in file:
                    img_list_L.append(img_path)
                    label_list_L.append(label_path)
                elif 'right' in file:
                    img_list_R.append(img_path)
                    label_list_R.append(label_path)
                else:
                    print('unknown problem')

    random_seed = 5
    np.random.seed(random_seed)
    np.random.shuffle(img_list_L)
    np.random.seed(random_seed)
    np.random.shuffle(label_list_L)
    np.random.seed(random_seed+1)
    np.random.shuffle(img_list_R)
    np.random.seed(random_seed+1)
    np.random.shuffle(label_list_R)

    test_num = min(len(img_list_L),len(img_list_R)) * test_per
    for i in range(len(img_list_L)):
        if i < test_num:
            test_img_L.write(img_list_L[i]+str('\n'))
            test_label_L.write(label_list_L[i]+str('\n'))
        else:
            train_img_L.write(img_list_L[i]+str('\n'))
            train_label_L.write(label_list_L[i]+str('\n'))
    for i in range(len(img_list_R)):
        if i < test_num:
            test_img_R.write(img_list_R[i]+str('\n'))
            test_label_R.write(label_list_R[i]+str('\n'))
        else:
            train_img_R.write(img_list_R[i]+str('\n'))
            train_label_R.write(label_list_R[i]+str('\n'))

    train_img_L.close()
    train_label_L.close()
    test_img_L.close()
    test_label_L.close()
    train_img_R.close()
    train_label_R.close()
    test_img_R.close()
    test_label_R.close()





def convert_label(label_dir,save_dir,map_dic):
    lut = np.zeros(256, dtype=np.uint8)
    for k in range(256):
        lut[k] = k
    for k in map_dic.keys():
        lut[k] = map_dic[k]

    for subdir, dirs, files in os.walk(label_dir):
        for file in files:
            if not '.png' in file:
                continue

            label_path = os.path.join(subdir, file)
            print(label_path)
            im = Image.open(label_path) # or load whatever ndarray you need
            im = np.array(im, dtype=np.uint8)
            print(im.shape)
            im = lut[im]

            if not os.path.exists(subdir.replace(label_dir,save_dir)):
                os.makedirs(subdir.replace(label_dir,save_dir))

            cv2.imwrite(label_path.replace(label_dir,save_dir),im)

def main():
    #img_dir = '/media/artur/Data_ssd/work/01_Project/BSD_weekly_training/data/20191024/bsd_night/origin/raw_img'
    #label_dir = '/media/artur/Data_ssd/work/01_Project/BSD_weekly_training/data/20191024/bsd_night/origin/gt_color'
    img_dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/2020-03-23/2020-03-20_toAndy0320-for-train-dataset/toAndy0320/images/night'
    label_dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/2020-03-23/2020-03-20_toAndy0320-for-train-dataset/toAndy0320/labels/night'
    #save_Dir = '/media/artur/Data_ssd/work/01_Project/BSD_weekly_training/data/20191024/bsd_night/'
    save_Dir = '/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/data/20200323/bsd_night'
    ## step1
    print('start move images')
    save_dir_img,save_dir_gtColor = move2sameFolder(img_dir, label_dir, save_Dir)

    # step2
    print('start convert to ID')
    save_dir_gtID = os.path.join(save_Dir,'gtID')
    cvtCol2ID.convert(save_dir_gtColor,save_dir_gtID)

    ## step3 convert complex ID into simple ID
    print('convert to simple ID')
    label_dic_info = {0:'empty',7:'road',24:'person',25:'rider',26:'car',27:'truck',28:'bus',33:'bicycle'}
    map_dic = {0:0,7:1,24:2,25:2,26:4,27:4,28:4,33:4}
    # map_dic = {0:0,7:1,24:2,25:2,26:3,27:4,28:5,33:6}
    save_dir_gtID_sp = os.path.join(save_Dir,'gtID_simple')
    convert_label(save_dir_gtID,save_dir_gtID_sp,map_dic)
    print('make list')
    test_per = 0.05
    make_list(test_per, save_dir_img, save_dir_gtID_sp, save_Dir)




if __name__ == '__main__':
    main()


