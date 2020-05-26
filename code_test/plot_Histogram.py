#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: plot_Histogram.py
@time: 5/14/20 4:07 PM
@purpose:draw Histogram
'''
import os
from numpy import array
import numpy as np
import pylab as pl
from matplotlib.ticker import FuncFormatter

# probability_file_false = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/2020/balace_probability/probability_false.txt'
# probability_file_true = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/2020/balace_probability/probability_true.txt'
# probability_file_no_ped = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/balace_test_probability/probability.txt'
balace_pro_path ='/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/balace_test_probability/probability.txt'
lower_pro_path ='/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/add_lower_feature_test_probability/probability.txt'
higher_pro_path = '/media/soterea/Data_ssd/work/YYZhang/test_file/experiment_results/5.13_lingshiceshi/temp/probability.txt'
def get_data(lines):
	sizeArr = []
	for line in lines:
		line =line.strip()
		print(line)
		line = float(line)
		sizeArr.append(line)
	return array(sizeArr)

# f_false = open(probability_file_false)
# f_true = open(probability_file_true)
# f__no_ped = open(probability_file_no_ped)
# balace_pro =open(balace_pro_path)
# lower_pro =open(lower_pro_path)



balace_ = open(balace_pro_path)
lower_ = open(lower_pro_path)
higher_ = open(higher_pro_path)




#
# lenths_false = get_data(f_false.readlines())
# lenths_true = get_data(f_true.readlines())
# lenths_no_ped = get_data(f__no_ped.readlines())
balace_pro =get_data(balace_.readlines())
lower_pro =get_data(lower_.readlines())
higher_pro = get_data(higher_.readlines())

def to_percent(y,position):
	return str(round(y,2)*100)+'%'
def draw_hist(lenths_true,lenths_false,lenths_no_ped):
	data_true = lenths_true
	data_false = lenths_false
	data_no_ped = lenths_no_ped
	bins = np.linspace(0,1,100)
	pl.hist(data_true,bins,weights=[1./len(data_true)]*len(data_true),facecolor = 'None',edgecolor = 'red',label='balace')
	pl.hist(data_false,bins,weights=[1./len(data_false)]*len(data_false),facecolor = 'None',edgecolor = 'black',label='lower')
	pl.hist(data_no_ped,bins,weights=[1./len(data_no_ped)]*len(data_no_ped),facecolor = 'None',edgecolor = 'green',label= 'higher')
	pl.legend()
	formater = FuncFormatter(to_percent)

	pl.xlabel('probability')
	pl.ylabel('the number of probability')
	pl.gca().yaxis.set_major_formatter(formater)
	pl.title('Predict ped probability Histogram on V_BSD_R_DN_SEG_02')
	pl.show()
def draw_hist_2(balace_pro,lower_pro,higher_pro):
	data_balace = balace_pro
	data_lower = lower_pro
	data_higher = higher_pro
	bins = np.linspace(0,1,100)
	pl.hist(data_balace,bins,facecolor = 'None',edgecolor = 'red',label='FP_on_temporarily_select_img_balace_model')
	pl.hist(data_lower,bins,facecolor = 'None',edgecolor = 'blue',label='FP_on_temporarily_select_img_lower_feature_fusion_model')
	pl.hist(data_higher,bins,facecolor = 'None',edgecolor = 'black',label='FP_on_temporarily_select_img_higher_feature_fusion_model')
	pl.xlabel('probability')
	pl.ylabel('Normalize the number of probability')
	pl.title('Predict ped probability Histogram on V_BSD_R_DN_SEG_02 and lower_fusion_model')
	pl.legend()
	pl.show()


# draw_hist(lenths_true)
# draw_hist(lenths_false)
# draw_hist(lenths_no_ped)
#draw_hist(balace_pro,lower_pro,higher_pro)
draw_hist_2(balace_pro,lower_pro,higher_pro)