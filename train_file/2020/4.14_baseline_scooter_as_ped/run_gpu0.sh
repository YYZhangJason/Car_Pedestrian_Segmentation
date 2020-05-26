#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-12-only-right-side/step_stage/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/01_Project/BSD_weekly_training/train/20191002/bench_weightUnion_oldLoss/BSD_Combi_day_191002_iter_60000.caffemodel" \
#--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-16-balance-examples/step_stage/train_log_step_stage

#/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
#--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020-3-12-only-right-side/l1reg/solver.prototxt" \
#--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/2020-3-16-balance-examples/step_stage/BSD_train_stage_one_d-n_iter_15000.caffemodel" \
#--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020-3-16-balance-examples/l1_reg/train_log_l1_reg


/media/soterea/Data_ssd/work/02_FrameWork/caffe-jacinto/build/tools/caffe.bin train \
--solver="/media/soterea/Data_ssd/work/YYZhang/train_file/2020/4.14_baseline_scooter_as_ped/sparse/solver.prototxt" \
--weights="/media/soterea/Data_ssd/work/YYZhang/Train_restore/base_line/model2/BSD_Combi_d-n_191217_iter_56000.caffemodel" \
--gpu=0 2>&1 | tee /media/soterea/Data_ssd/work/YYZhang/log_path/2020/sparse/train_log_sparse
