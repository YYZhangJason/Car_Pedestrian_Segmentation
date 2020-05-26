#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/step_stage/solver.prototxt" \
##--weights="/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/train/20191002/bench_weightUnion_oldLoss/BSD_Combi_day_191002_iter_60000.caffemodel" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel" \
#--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/step_stage/train_log_step_stage
#
/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/l1reg/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/step_stage/BSD_train_stage_one_d-n_iter_80000.caffemodel" \
--gpu=1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/l1_reg/train_log_l1_reg

#
#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-4-train_out_5/sparse/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel" \
#--gpu=1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-4-train_4xaugmentation_change_loss_raito_dilation/sparse/train_log_sparse
