
for SPLIT in 0 1 2 3 4; do
for SIZE in 64 128 256; do
for CONFIG in fcn_unet_s5-d16_${SIZE}x${SIZE}_40k_gd_line512new_lr0.01 deeplabv3_unet_s5-d16_${SIZE}x${SIZE}_40k_gd_line512new_lr0.01 pspnet_unet_s5-d16_${SIZE}x${SIZE}_40k_gd_line512new_lr0.01; do

WORK_DIR=/media/ubuntu/SSD/mmsegmentation/RevisionV1/${SPLIT}/${CONFIG}

if [ ! -d ${WORK_DIR} ]; then

echo "${WORK_DIR} not existed"

./tools/dist_train.sh configs/unet/${CONFIG}.py 1 --work-dir ${WORK_DIR}

sleep 60

fi

done
done
done



















