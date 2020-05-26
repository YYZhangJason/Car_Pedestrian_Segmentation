/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020/step_stage/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/base_line/model2/BSD_Combi_d-n_191217_iter_56000.caffemodel" \
--gpu "1" 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020/step_stage/train_log_first_stage
#
#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/train_data_augmentation/l1reg/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/step_stage/BSD_train_stage_one_d-n_iter_80000.caffemodel" \
#--gpu "1" 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-2-21/l1_reg/train_log_l1reg


#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/train_reduce_ped_loss/sparse/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/l1_reg/BSD_train_l1reg_d-n_iter_30000.caffemodel" \
#--gpu "1" 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-2-21/sparse/train_log_sparse
