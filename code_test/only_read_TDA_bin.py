#!/usr/bin/env python3
# encoding: utf-8
'''
@author: YYZhang
@file: only_read_TDA_bin.py
@time: 3/30/20 7:22 PM
@purpose:
'''
def get_TDA_array(path, height, width, ):
	fp = open(path, 'rb')
	frame_len = height * width  # 一帧图像所含的像素个数

	fp.seek(0, 2)  # 设置文件指针到文件流的尾部
	ps = fp.tell()  # 当前文件指针位置
	numfrm = ps // frame_len  # 计算输出帧数
	fp.seek(0, 0)

	return fp, numfrm
def main():
	tda_file = '/media/soterea/Data_ssd/work/YYZhang/windows/test/2020/stats_tool_out.bin'
	fp, numfrm = get_TDA_array(tda_file,720,360)
	print(numfrm)
if __name__ == '__main__':
	main()