import os

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import cv2
from datetime import datetime
import sys
CAFFE_HOME = '/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_HOME + 'python')
import caffe
test_file = '/media/soterea/Data_ssd/work/YYZhang/test_file/test_data/2020-2-15/record.csv'
save_path = ''
test_prototxt = '/media/soterea/Data_ssd/work/YYZhang/train_file/train_data_augmentation/sparse/deploy.prototxt'
weights = '/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-19_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel'
