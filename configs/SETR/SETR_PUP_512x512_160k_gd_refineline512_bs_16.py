# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/refine_line_v1_512_512'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
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
        img_scale=(2048, 512),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/annotations',
        split='train.txt',
        pipeline=train_pipeline),
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

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformerInSETR',
        model_name='vit_large_patch16_384',
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=2,
        drop_rate=0.,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
    ),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=512,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=4,
        num_upsampe_layer=4,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=9,
        img_size=512,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=14,
            img_size=512,
            embed_dim=1024,
            num_classes=2,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            num_upsampe_layer=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=19,
            img_size=512,
            embed_dim=1024,
            num_classes=2,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            num_upsampe_layer=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=23,
            img_size=512,
            embed_dim=1024,
            num_classes=2,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            num_upsampe_layer=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg = dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

# optimizer
# bs=16
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
#                  paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.000125,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
total_iters = 160000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=2000, metric='mIoU')

find_unused_parameters = True
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
