#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/normal/step_stage/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/train/20191002/bench_weightUnion_oldLoss/BSD_Combi_day_191002_iter_60000.caffemodel" \
#--gpu=0,1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-2-26/step_stage/train_log_first_stage
#
#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/normal/l1reg/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-19_train/step_stage/BSD_train_stage_one_d-n_iter_80000.caffemodel" \
#--gpu=0,1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-2-26/l1_reg/train_log_l1reg
#

/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/train_reduce_dropout_3_times_augumentation/sparse/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-19_train/l1_reg/BSD_train_l1reg_d-n_iter_30000.caffemodel" \
--gpu "0" 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-2-26-train_reduce_dropout_3_times_augumentation/sparse/train_log_sparse
