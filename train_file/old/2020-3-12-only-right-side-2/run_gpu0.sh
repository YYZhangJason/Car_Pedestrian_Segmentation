#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-12-only-right-side-2/step_stage/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/train/20191002/bench_weightUnion_oldLoss/BSD_Combi_day_191002_iter_60000.caffemodel" \
##--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-2-21_train/sparse/BSD_train_sparse_d-n_iter_60000.caffemodel" \
#--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-12-only-right-side-2/step_stage

#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-12-only-right-side-2/l1reg/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-12-only-right-side-2/step_stage/BSD_train_stage_one_d-n_iter_60000.caffemodel" \
#
#--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-12-only-right-side-2/l1_reg/train_log_l1_reg
#

/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-12-only-right-side-2/sparse/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-12-only-right-side-2/sparse/old/BSD_train_sparse_d-n_iter_60000.caffemodel" \
--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-12-only-right-side-2/sparse/train_log_sparse
