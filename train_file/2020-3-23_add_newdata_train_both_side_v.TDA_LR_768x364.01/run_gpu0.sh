/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/step_stage/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/train/20191002/bench_weightUnion_oldLoss/BSD_Combi_day_191002_iter_60000.caffemodel" \
--gpu=1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/step_stage/train_log_first_stage

/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/l1reg/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/step_stage/BSD_train_stage_one_d-n_iter_40000.caffemodel" \
--gpu=1 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/l1_reg/train_log_l1reg


/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/sparse/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/l1_reg/BSD_train_l1reg_d-n_iter_30000.caffemodel" \
--gpu=1  2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-23_add_newdata_train_both_side_v.TDA_LR_768x364.01/sparse/train_log_sparse
