_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/gd_mc3classes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=3), auxiliary_head=dict(num_classes=3))
evaluation = dict(interval=800001, metric='mIoU')