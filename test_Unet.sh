





for ARCH in fcn_unet_s5-d16 deeplabv3_unet_s5-d16 pspnet_unet_s5-d16; do

  for SIZE in 64x64 128x128 256x256; do

    CONFIG=${ARCH}_${SIZE}_40k_gd_line512new_lr0.01
    echo $CONFIG

    CUDA_VISIBLE_DEVICES=1 python tools/test_modified.py \
    /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
    /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
    --work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
    --eval mDice mIoU mFscore \
    --out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/unet_s5-d16_256x256_40k_line512new_lr0.01/result_${ARCH}_${SIZE}.pkl

  done

done
