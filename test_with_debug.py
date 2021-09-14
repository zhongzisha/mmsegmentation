# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys,os
import os.path as osp
import shutil
import time
import warnings
import tempfile
import warnings

import cv2
import mmcv
import numpy as np

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

# from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.image import tensor2imgs



"""
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.01_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.01.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnet_0.3.pkl

#########################################################################################################

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV2_0.3.pkl


#######################################################################################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V3_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV3_0.3.pkl

############ 2 layers ###########################################################################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.05_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.1_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.2_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.3_tiny_224_4_/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_0.3.pkl

############ 2 layers (END) ############################################################################################



############ 3 layers ###################################
CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.05_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.05.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.1_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.1.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.2_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.2.pkl

CONFIG=unet_s5-d16_256x256_40k_line512new_lr0.01
python test_with_debug.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/${CONFIG}.py \
/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/iter_40000.pth \
--work-dir /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/ \
--eval mDice mIoU \
--other_results_dir /media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V4_bs=24_lr=0.3_tiny_224_4_3layers/predictions \
--out /media/ubuntu/Temp/gd/mmsegmentation/work_dirs/${CONFIG}/result_SwinUnetV4_3layers_0.3.pkl

############ 3 layers (END) ################################################################################
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--other_results_dir', type=str, default='')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    return_results=True,
                    args=None):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result2 = model(return_loss=False, **data)

        # import pdb
        # pdb.set_trace()
        #  to get the filename, use data['img_metas'][0].data[0][0]['filename']
        filename = data['img_metas'][0].data[0][0]['ori_filename']
        # '/media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_line_seg_Swin_Unet_bs=24_lr=0.05_withSat/predictions'
        results_dir = args.other_results_dir
        pred = cv2.imread(os.path.join(results_dir,
                                       filename.replace('.jpg', '.png')))
        pred = pred[:, (1024):(1024+512), 0]
        # pdb.set_trace()
        pred[pred > 0] = 1
        result = [pred.astype(np.int64)]

        # pdb.set_trace()

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if return_results:
            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result, tmpdir='.efficient_test')
                results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


# Record the information printed in the terminal
class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create

    pkl_filename = args.out
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
        json_file = osp.join(args.work_dir, os.path.basename(pkl_filename).replace('.pkl', '.json'))
        sys.stdout = Print_Logger(osp.join(args.work_dir, os.path.basename(pkl_filename).replace('.pkl', '_log.txt')))

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs,
            return_results=True,
            args=args)
    else:
        pass
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False)
        # results = multi_gpu_test(
        #     model,
        #     data_loader,
        #     args.tmpdir,
        #     args.gpu_collect,
        #     False,
        #     pre_eval=args.eval is not None and not eval_on_format_results,
        #     format_only=args.format_only or eval_on_format_results,
        #     format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            metric = dataset.evaluate(results, args.eval, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
