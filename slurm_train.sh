#!/bin/bash
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=g39
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/mmseg-job-train-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/mmseg-job-train-%j-%N.err

echo "job start `date`"
echo "job run at ${HOSTNAME}"
nvidia-smi

df -h
nvidia-smi
ls /usr/local
which nvcc
which gcc
which g++
nvcc --version
gcc --version
g++ --version

env

nvidia-smi

free -g
top -b -n 1

uname -a

#sleep 50000

if [ ${HOSTNAME} == "g38" ]; then
  source $HOME/anaconda3_py38/bin/activate   # here, opencv-4.5.2

  CONFIG=configs/SETR/SETR_MLA_512x512_160k_gd_refineline512_bs_16.py
  # CONFIG=deeplabv3_unet_s5-d16_256x256_40k_gd_refineline512
  echo $CONFIG
  CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG} 1 || exit

fi

if [ ${HOSTNAME} == "g39" ]; then
  source $HOME/anaconda3_py38/bin/activate   # here, opencv-4.5.2

  CONFIG=configs/unet/deeplabv3_unet_s5-d16_256x256_40k_gd_refineline512.py
  CONFIG=configs/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1.py
  CONFIG=configs/SETR/SETR_MLA_256x256_160k_gd_refineline512_bs_16.py
  echo $CONFIG

  CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG} 1 || exit

fi

if [ ${HOSTNAME} == "g42" ]; then
  source $HOME/anaconda3_py38/bin/activate   # here, opencv-4.5.2

  CONFIG=configs/unet/fcn_unet_s5-d16_512x512_40k_gd_refineline512.py
  CONFIG=configs/unet/fcn_unet_s5-d16_256x256_40k_gd_refineline512.py
  echo $CONFIG

  CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG} 1 || exit

fi

if [ ${HOSTNAME} == "g41" ]; then
  source $HOME/anaconda3_py38/bin/activate   # here, opencv-4.5.2

  CONFIG=configs/unet/pspnet_unet_s5-d16_512x512_40k_gd_refineline512.py
  #CONFIG=configs/unet/pspnet_unet_s5-d16_256x256_40k_gd_refineline512.py
  echo $CONFIG

  CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG} 1 || exit

fi


