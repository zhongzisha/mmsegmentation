_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/gd_mc4classes_v6.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
optimizer = dict(lr=0.001)
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=20000, metric='mIoU')
model = dict(
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5))