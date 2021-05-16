norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (256, 256)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VIT_MLA',
        model_name='vit_large_patch16_384',
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=19,
        drop_rate=0.0,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(5, 11, 17, 23)
    ),
    decode_head=dict(
        type='VIT_MLAHead',
        in_channels=1024,
        channels=512,
        img_size=256,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=256,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=256,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=256,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=256,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(256, 256), stride=(170, 170))
)

dataset_type = 'CustomDataset'
data_root = 'data/refine_line_v1_512_512'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=160000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/annotations',
            split='train.txt',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/annotations',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/annotations',
        split='val.txt',
        pipeline=test_pipeline))

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
total_iters = 160000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=1000, metric='mIoU')

find_unused_parameters = True
