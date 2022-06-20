LC_TIME=en_US.UTF-8

# shellcheck disable=SC2006
time1=`date`              # 获取当前时间
time2=$(date -d "-240 minute ago" +"%Y-%m-%d %H:%M:%S")  # 获取两个小时后的时间

t1=`date -d "$time1" +%s`     # 时间转换成timestamp
t2=`date -d "$time2" +%s`

echo t1=$t1
echo t2=$t2

while [ $t1 -lt $t2 ]     # 循环，不断检查是否来到了未来时间
do
  echo "wait for 60 seconds .."
  sleep 60
  time1=`date`
  t1=`date -d "$time1" +%s`
  echo t1=$t1
done

echo "yes"       # 循环结束，开始执行任务
echo $time1
echo $time2

sleep 60


CONFIG=deeplabv3_unet_s5-d16_128x128_40k_gd_line512new_lr0.01
./tools/dist_train.sh configs/unet/${CONFIG}.py 1 \
--work-dir /media/ubuntu/Temp/mmsegmentation/${CONFIG}

sleep 60

CONFIG=fcn_unet_s5-d16_256x256_40k_line512new_lr0.01
./tools/dist_train.sh configs/unet/${CONFIG}.py 1 \
--work-dir /media/ubuntu/Temp/mmsegmentation/${CONFIG}

sleep 60

CONFIG=pspnet_unet_s5-d16_256x256_40k_gd_line512new_lr0.01
./tools/dist_train.sh configs/unet/${CONFIG}.py 1 \
--work-dir /media/ubuntu/Temp/mmsegmentation/${CONFIG}





















