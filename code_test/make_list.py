#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: make_list.py
@time: 4/14/20 11:15 AM
@purpose:生成训练文件和测试文件
'''
import numpy as np
import os
def make_list(test_per,img_label_parent_path,save_Dir):

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
    for dir_img in os.listdir(img_label_parent_path):
        dir_img_path = os.path.join(img_label_parent_path,dir_img)
        for list_day_night in os.listdir(dir_img_path):
            day_night_path = os.path.join(dir_img_path,list_day_night)
            raw_path = os.path.join(day_night_path,'raw_img')
            label_path = os.path.join(day_night_path,'gtID_simple')
            for raw_path_left_right in os.listdir(raw_path):
                LR_path = os.path.join(raw_path,raw_path_left_right)
                for img_name in os.listdir(LR_path):
                    label_name = os.path.join(label_path,img_name)
                    raw_img_name = os.path.join(LR_path,img_name)
                    if not os.path.exists(label_name):
                        print('!warmming:', label_name)
                    else:
                        if 'left' in raw_img_name:
                            img_list_L.append(raw_img_name)
                            label_list_L.append(label_name)
                        elif 'right' in raw_img_name:
                                img_list_R.append(raw_img_name)
                                label_list_R.append(label_name)
                        else:
                            print("not exist raw img or label img!")

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
if __name__ == '__main__':
    test_per = 0.1
    img_label_parent_path = '/media/soterea/Data_ssd/work/YYZhang/data/train_data'
    save_dir = '/media/soterea/Data_ssd/work/YYZhang/data/train_test_txt'
    make_list(test_per,img_label_parent_path,save_dir)