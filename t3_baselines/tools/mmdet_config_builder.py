from pathlib import Path


CLASS_NAMES_10 = (
    "Exterior urban spaces with people",
    "Exterior urban spaces without people",
    "Interior urban spaces with people",
    "Interior urban spaces without people",
    "Hotel or commercial lodging spaces",
    "Private home interiors",
    "Food or drink items",
    "Retail products and merchandise",
    "Human-centered portrait",
    "Other non-spatial content",
)


def build_mmdet_config(
    model,
    cfg_path,
    data_root,
    img_root,
    work_dir,
    num_classes,
    batch_size,
    epochs,
):
    cfg_path = Path(cfg_path)

    if model == "cascade_mask_rcnn":
        cfg_text = build_cascade_mask_rcnn_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
        )
    elif model == "solov2":
        cfg_text = build_solov2_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
        )
    elif model == "mask2former":
        cfg_text = build_mask2former_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg_text, encoding="utf-8")


def _metainfo(num_classes):
    if num_classes == 10:
        return f"dict(classes={CLASS_NAMES_10})"
    names = tuple(str(i) for i in range(num_classes))
    return f"dict(classes={names})"


def common_dataset_cfg(data_root, img_root, num_classes, batch_size):
    metainfo = _metainfo(num_classes)

    return f"""
dataset_type = 'CocoDataset'
data_root = r'{data_root}'
img_root = r'{img_root}'
metainfo = {metainfo}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=img_root + '/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=img_root + '/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=img_root + '/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args
)
"""


def common_runtime_cfg(work_dir, epochs):
    return f"""
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval=1)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end={epochs}, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05)
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = r'{work_dir}'
"""


def build_cascade_mask_rcnn_cfg(data_root, img_root, work_dir, num_classes, batch_size, epochs):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, batch_size)
    runtime_cfg = common_runtime_cfg(work_dir, epochs)

    return f"""
_base_ = 'mmdet::_base_/models/cascade-mask-rcnn_r50_fpn.py'

{dataset_cfg}

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes={num_classes}),
            dict(type='Shared2FCBBoxHead', num_classes={num_classes}),
            dict(type='Shared2FCBBoxHead', num_classes={num_classes})
        ],
        mask_head=dict(num_classes={num_classes})
    )
)

{runtime_cfg}
"""


def build_solov2_cfg(data_root, img_root, work_dir, num_classes, batch_size, epochs):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, batch_size)
    runtime_cfg = common_runtime_cfg(work_dir, epochs)

    return f"""
_base_ = 'mmdet::solov2/solov2_r50_fpn_1x_coco.py'

{dataset_cfg}

model = dict(
    mask_head=dict(num_classes={num_classes})
)

{runtime_cfg}
"""


def build_mask2former_cfg(data_root, img_root, work_dir, num_classes, batch_size, epochs):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, batch_size)
    runtime_cfg = common_runtime_cfg(work_dir, epochs)

    return f"""
_base_ = 'mmdet::mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'

{dataset_cfg}

model = dict(
    panoptic_head=dict(
        num_things_classes={num_classes},
        num_stuff_classes=0,

        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=[1.0] * ({num_classes} + 1)
        )
    ),

    panoptic_fusion_head=dict(
        num_things_classes={num_classes},
        num_stuff_classes=0
    )
)

{runtime_cfg}
"""