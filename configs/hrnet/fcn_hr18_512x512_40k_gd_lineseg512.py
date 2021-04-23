_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/gd_lineseg512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=2))
