current_dir=`pwd`

cd /media/ubuntu/Documents/gd || exit

python extract_gt_patches_forAug.py --aug_times 1 --aug_type mc_seg_v7 --subset train --random_count 8

python extract_gt_patches_forAug.py --aug_times 1 --aug_type mc_seg_v7 --subset val --random_count 8

cd $current_dir || exit