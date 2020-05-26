#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: plot_train_img.py
@time: 2/29/20 12:11 PM
@desc:
'''
# -*- coding: utf-8 -*-
'''
读取Caffe Log文件 并进行图表显示

'''
import re
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
# Kai = matplotlib.font_manager.FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')  # 设置中文


def moving_avg(x,y,n):
    mode = 'valid'
    x = x[n//2:-(n//2)]
    y = np.convolve(y, np.ones((n,)) / n, mode=mode)
    return x,y



def read_log_file(url,cls_dic):
    file = open(url, 'r')

    loss_test = []
    loss_train = []
    iou_train = {}
    acc_train = {}
    iou_test = {}
    acc_test = {}

    for line in file:

        ## get train loss & iter
        if re.search('Iteration \d*? \(',line):
            s = re.search('Iteration \d*? \(', line)
            iter_train = int(line[s.start()+len('Iteration'):s.end()-1])
            # print('iter_train',iter_train)
        elif re.search('Train net output #0:',line):
            s = re.search('loss = .*? \(', line)
            loss = float(line[s.start()+len('loss = '):s.end()-1])
            loss_train.append([iter_train,loss])
            # print(line[s.start():s.end()-1])


        ## get test loss & iter
        elif re.search('Iteration \d*?, Testing net', line):
            s = re.search('Iteration \d*?,', line)
            iter_test = int(line[s.start() + len('Iteration'):s.end() - 1])
            # print('iter_test', iter_test)
        elif re.search('Test net output #0:', line):
            s = re.search('loss = .*? \(', line)
            loss = float(line[s.start()+len('loss = '):s.end()-1])
            loss_test.append([iter_test,loss])
            # print(line[s.start():s.end() - 1])


        ## get IOU & acc
        elif re.search('current iteration', line):
            if re.search('TEST', line):
                iou_flag = 'test'
            elif re.search('TRAIN', line):
                iou_flag = 'train'
                s = re.search('iteration-train:.*?', line)
                iter_train = int(line[s.start()+len('iteration-train:'):])
        elif re.search('Accuracy:', line):
            s = line.split('\t')
            if iou_flag == 'train':
                for id in cls_dic:
                    if cls_dic[id] not in acc_train:
                        acc_train[cls_dic[id]] = []
                    if float(s[id + 1]) != -1:
                        acc_train[cls_dic[id]].append([iter_train,float(s[id+1])])
            elif iou_flag == 'test':
                for id in cls_dic:
                    if cls_dic[id] not in acc_test:
                        acc_test[cls_dic[id]] = []
                    acc_test[cls_dic[id]].append([iter_test,float(s[id+1])])
        elif re.search('Iou:', line):
            s = line.split('\t')
            print(s)
            if iou_flag == 'train':
                for id in cls_dic:
                    if cls_dic[id] not in iou_train:
                        iou_train[cls_dic[id]] = []
                    if float(s[id + 1]) != -1:
                        iou_train[cls_dic[id]].append([iter_train, float(s[id + 1])])
            elif iou_flag == 'test':
                for id in cls_dic:
                    if cls_dic[id] not in iou_test:

                        iou_test[cls_dic[id]] = []
                    iou_test[cls_dic[id]].append([iter_test, float(s[id + 1])])
                    print(id)
                    print(s[id])


    file.close()

    return loss_train,loss_test,iou_train,acc_train,iou_test,acc_test


def main(url,cls_dic,title):
    #fig_save_dir = os.path.join(os.path.dirname(url),'Training_plot')
    fig_save_dir = os.path.join('/media/soterea/Data_ssd/work/YYZhang/log_path/2020/sparse/','Training_plot')
    print(fig_save_dir)
    if  not os.path.exists(fig_save_dir):

        os.mkdir(fig_save_dir)

    loss_train,loss_test,iou_train,acc_train,iou_test,acc_test =read_log_file(url,cls_dic)

    ## fig1 loss
    loss_test = np.asarray(loss_test)
    loss_train = np.asarray(loss_train)

    plt.figure(1,figsize=(16,10))
    plt.suptitle(title,fontsize='x-large')
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    plt.plot(loss_train[:,0],loss_train[:,1])
    plt.title('loss_train_baseline_jpeg')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    # plt.ylim(0, 1)
    plt.sca(ax1)

    plt.plot(loss_test[:,0],loss_test[:,1])
    plt.title('loss_test_baseline_jpeg')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    # plt.ylim(0, 1)
    plt.sca(ax2)

    plt.savefig(os.path.join(fig_save_dir,title+'_loss.png'))

    ## fig2 acc
    plt.figure(2,figsize=(16,10))
    plt.suptitle(title,fontsize='x-large')
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    for i in [0,1,2,4]:
    #for i in [0,2,4]:
        cls = cls_dic[i]
        acc_cls_train = np.asarray(acc_train[cls])
        print(acc_cls_train)
        xnew = np.asarray(acc_cls_train[:,0])

        ynew = np.asarray(acc_cls_train[:,1])
        xnew,ynew = moving_avg(xnew,ynew,n=15)

        # plt.plot(acc_cls_train[:,0],acc_cls_train[:,1])
        plt.plot(xnew,ynew,label=cls)
        plt.title('Class ACC_Train_baseline_jpeg')
        plt.xlabel('Iteration')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))

        acc_cls_test = np.asarray(acc_test[cls])
        xnew = np.asarray(acc_cls_test[:,0])
        ynew = np.asarray(acc_cls_test[:,1])
        # xnew,ynew = moving_avg(xnew,ynew,n=3)

        # plt.plot(acc_cls_test[:,0],acc_cls_test[:,1])
        plt.plot(xnew,ynew)
        plt.title('Class ACC_Test_baseline_jpeg')
        plt.xlabel('Iteration')
        plt.ylabel('ACC')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax2)
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.legend()
    plt.savefig(os.path.join(fig_save_dir,title+'_acc.png'))

    ## fig3 iou
    plt.figure(3,figsize=(16,10))
    plt.suptitle(title,fontsize='x-large')
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    #for i in [0,2,4]:
    for i in [0,1,2,4]:
        cls = cls_dic[i]
        iou_cls_train = np.asarray(iou_train[cls])
        xnew = np.asarray(iou_cls_train[:,0])
        ynew = np.asarray(iou_cls_train[:,1])
        xnew,ynew = moving_avg(xnew,ynew,n=15)

        # plt.plot(acc_cls_train[:,0],acc_cls_train[:,1])
        plt.plot(xnew,ynew,label=cls)
        plt.title('Class IOU_Train_baseline_jpeg')
        plt.xlabel('Iteration')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))

        iou_cls_test = np.asarray(iou_test[cls])
        xnew = np.asarray(iou_cls_test[:,0])
        ynew = np.asarray(iou_cls_test[:,1])

        # xnew,ynew = moving_avg(xnew,ynew,n=3)

        # plt.plot(acc_cls_test[:,0],acc_cls_test[:,1])
        plt.plot(xnew,ynew)
        plt.title('Class IOU_Test_baseline_jpeg')
        plt.xlabel('Iteration')
        plt.ylabel('IOU')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax2)
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.legend()
    plt.savefig(os.path.join(fig_save_dir,title+'_iou.png'))



    ## fig4 score
    plt.figure(4,figsize=(16,10))
    plt.suptitle(title,fontsize='x-large')
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    for i in [2,4]:
        cls = cls_dic[i]
        iou_cls_train = np.asarray(iou_train[cls])
        acc_cls_train = np.asarray(acc_train[cls])
        print (iou_cls_train)
        print(acc_cls_train)
        xnew = np.asarray(acc_cls_train[:,0])
        if len(iou_cls_train)>len(acc_cls_train):
            ynew = []
            ynew_1 = 0.6*np.asarray(iou_cls_train[:,1])/np.max(np.asarray(iou_cls_train[:,1]))
            ynew_2 = 0.4*np.asarray(acc_cls_train[:,1])/np.max(np.asarray(acc_cls_train[:,1]))
            for i,x_ in enumerate(xnew):
                if x_ in iou_cls_train[:,0]:
                    ynew.append(ynew_1[i]+ynew_2[i])
        else:
            ynew = 0.6*np.asarray(iou_cls_train[:,1])/np.max(np.asarray(iou_cls_train[:,1]))+\
            0.4*np.asarray(acc_cls_train[:,1])/np.max(np.asarray(acc_cls_train[:,1]))
        xnew,ynew = moving_avg(xnew,ynew,n=15)

        # plt.plot(acc_cls_train[:,0],acc_cls_train[:,1])
        plt.plot(xnew,ynew,label=cls)
        plt.title('Class Score_Train')
        plt.xlabel('Iteration')
        # plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax1)
        # ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        # ax1.yaxis.set_minor_locator(MultipleLocator(0.05))

        iou_cls_test = np.asarray(iou_test[cls])
        acc_cls_test = np.asarray(acc_test[cls])
        xnew = np.asarray(iou_cls_test[:,0])
        ynew = 0.5*np.asarray(iou_cls_test[:,1])/np.max(np.asarray(iou_cls_test[:,1])) + \
               0.5*np.asarray(np.asarray(acc_cls_test[:,1]))/np.max(np.asarray(acc_cls_test[:,1]))
        # xnew,ynew = moving_avg(xnew,ynew,n=3)

        # plt.plot(acc_cls_test[:,0],acc_cls_test[:,1])
        plt.plot(xnew,ynew)
        plt.title('Class Score_Test')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        # plt.ylim(0, 1)
        plt.grid(True)
        plt.sca(ax2)
        # ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        # ax2.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.legend()
    plt.savefig(os.path.join(fig_save_dir,title+'_score.png'))

    plt.show()


if __name__ == '__main__':
    log_file = '/media/soterea/Data_ssd/work/YYZhang/log_path/2020/sparse/train_log_sparse'
    title = 'test'
    #cls_dic ={0:'Empty',2:'Ped',3:'unused',4:'Car'}
    cls_dic ={0:'Empty',1:'Road',2:'Ped',3:'unused',4:'Car'}
    main(log_file,cls_dic,title)