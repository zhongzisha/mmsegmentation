




for SPLIT in {0..4}; do
for ARCH in fcn_unet_s5-d16 deeplabv3_unet_s5-d16 pspnet_unet_s5-d16; do

  for SIZE in 64x64 128x128 256x256; do

    CONFIG=${ARCH}_${SIZE}_40k_gd_line512new_lr0.01
    echo $CONFIG

    CUDA_VISIBLE_DEVICES=0 python tools/test_modified.py \
    /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/${CONFIG}.py \
    /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/iter_40000.pth \
    --work-dir /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}/ \
    --eval mDice mIoU mFscore \
    --out /media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/result_${ARCH}_${SIZE}.pkl

  done

done

done