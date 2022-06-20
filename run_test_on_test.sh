
CONFIG=fcn_unet_s5-d16_256x256_40k_gd_line512new_lr0.01
CONFIG=deeplabv3_unet_s5-d16_256x256_40k_gd_line512new_lr0.01
CONFIG=pspnet_unet_s5-d16_256x256_40k_gd_line512new_lr0.01
python tools/test.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}_test.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--show-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/test

for CONFIG in fcn_unet_s5-d16_256x256_40k_gd_line512new_lr0.01 deeplabv3_unet_s5-d16_256x256_40k_gd_line512new_lr0.01 pspnet_unet_s5-d16_256x256_40k_gd_line512new_lr0.01; do
SUBSET=val
python tools/test.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}_${SUBSET}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--show-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${SUBSET}


done

