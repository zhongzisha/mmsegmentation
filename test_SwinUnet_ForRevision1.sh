CONFIG=fcn_unet_s5-d16_64x64_40k_gd_line512new_lr0.01

for SPLIT in {0..4}; do

for LR in 0.05 0.1 0.2; do

python test_with_debug.py \
/media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/${CONFIG}.py \
/media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/SSD/VesselSeg-Pytorch/gd/RevisionV1/gd_lineseg_new_Swin_Unet_bs=24_lr=${LR}_tiny_224_4_/val \
--out /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/result_SwinUnet_${LR}.pkl

done
done

exit

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.3.pkl

#########################################################################################################

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.3.pkl


#######################################################################################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.3.pkl

############ 2 layers ###########################################################################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.3.pkl

############ 2 layers (END) ############################################################################################



############ 3 layers ###################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.05_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.1_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.2_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU mFscore \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.3_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.3.pkl

############ 3 layers (END) ################################################################################