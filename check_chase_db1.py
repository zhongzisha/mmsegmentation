import sys,os,glob
import numpy as np
import cv2

subset = sys.argv[1]
img_dir = os.path.join('data/CHASE_DB1/images/%s' % subset)
ann_dir = os.path.join('data/CHASE_DB1/annotations/%s' % subset)
image_files = glob.glob(img_dir + '/*.png')
for img_filename in image_files:
    prefix = img_filename.split(os.sep)[-1].replace('.png', '')
    ann_filename = ann_dir + '/' + prefix + '_1stHO.png'
    img = cv2.imread(img_filename)
    gt = cv2.imread(ann_filename)
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]

    print(prefix, img.shape)
    print('gt_min:', gt.min())
    print('gt_max:', gt.max())
    print('gt_unique:', np.unique(gt))



