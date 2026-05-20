from pathlib import Path


def build_mmdet_config(
    model,
    cfg_path,
    data_root,
    img_root,
    work_dir,
    num_classes,
    class_names,
    batch_size,
    epochs,
    lr=0.0002,
    warmup_iters=50,
    num_workers=2,
    load_from=None,
):
    cfg_path = Path(cfg_path)

    if model == "cascade_mask_rcnn":
        cfg_text = build_cascade_mask_rcnn_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            warmup_iters=warmup_iters,
            num_workers=num_workers,
            load_from=load_from,
        )
    elif model == "mask_rcnn":
        cfg_text = build_mask_rcnn_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            warmup_iters=warmup_iters,
            num_workers=num_workers,
            load_from=load_from,
        )
    elif model == "solov2":
        cfg_text = build_solov2_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            warmup_iters=warmup_iters,
            num_workers=num_workers,
            load_from=load_from,
        )
    elif model == "mask2former":
        cfg_text = build_mask2former_cfg(
            data_root=data_root,
            img_root=img_root,
            work_dir=work_dir,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            warmup_iters=warmup_iters,
            num_workers=num_workers,
            load_from=load_from,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg_text, encoding="utf-8")


def _metainfo(num_classes, class_names=None):
    if class_names:
        names = tuple(class_names)
    else:
        names = tuple(str(i) for i in range(num_classes))
    return f"dict(classes={names})"


def common_dataset_cfg(data_root, img_root, num_classes, class_names, batch_size, num_workers):
    metainfo = _metainfo(num_classes, class_names)
    persistent_workers = num_workers > 0

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
    num_workers={num_workers},
    persistent_workers={persistent_workers},
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
    num_workers={num_workers},
    persistent_workers={persistent_workers},
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
    num_workers={num_workers},
    persistent_workers={persistent_workers},
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


def common_runtime_cfg(work_dir, epochs, lr=0.0002, warmup_iters=50, load_from=None, delete_base=True):
    milestone1 = max(1, int(epochs * 0.7))
    milestone2 = max(milestone1 + 1, int(epochs * 0.9))
    if milestone2 >= epochs:
        milestone2 = max(1, epochs - 1)
    load_from_value = "None" if not load_from else f"r'{load_from}'"
    delete_kw = "_delete_=True, " if delete_base else ""
    optim_delete_line = "    _delete_=True,\n" if delete_base else ""
    checkpoint_delete_kw = "_delete_=True, " if delete_base else ""
    return f"""
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict({checkpoint_delete_kw}type='CheckpointHook', interval=1, by_epoch=True, max_keep_ckpts=2),
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
load_from = {load_from_value}
resume = False

train_cfg = dict({delete_kw}type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval=1)
val_cfg = dict({delete_kw}type='ValLoop')
test_cfg = dict({delete_kw}type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end={warmup_iters}),
    dict(type='MultiStepLR', begin=0, end={epochs}, by_epoch=True, milestones=[{milestone1}, {milestone2}], gamma=0.1)
]

optim_wrapper = dict(
{optim_delete_line}\
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr={lr}, weight_decay=0.05)
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = r'{work_dir}'
"""


def build_cascade_mask_rcnn_cfg(data_root, img_root, work_dir, num_classes, class_names, batch_size, epochs, lr, warmup_iters, num_workers, load_from):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, class_names, batch_size, num_workers)
    runtime_cfg = common_runtime_cfg(work_dir, epochs, lr, warmup_iters, load_from, delete_base=False)

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


def build_mask_rcnn_cfg(data_root, img_root, work_dir, num_classes, class_names, batch_size, epochs, lr, warmup_iters, num_workers, load_from):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, class_names, batch_size, num_workers)
    runtime_cfg = common_runtime_cfg(work_dir, epochs, lr, warmup_iters, load_from, delete_base=False)

    return f"""
_base_ = 'mmdet::_base_/models/mask-rcnn_r50_fpn.py'

{dataset_cfg}

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes={num_classes}),
        mask_head=dict(num_classes={num_classes})
    )
)

{runtime_cfg}
"""


def build_solov2_cfg(data_root, img_root, work_dir, num_classes, class_names, batch_size, epochs, lr, warmup_iters, num_workers, load_from):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, class_names, batch_size, num_workers)
    runtime_cfg = common_runtime_cfg(work_dir, epochs, lr, warmup_iters, load_from)

    return f"""
_base_ = 'mmdet::solov2/solov2_r50_fpn_1x_coco.py'

{dataset_cfg}

model = dict(
    mask_head=dict(num_classes={num_classes}),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.01,
        mask_thr=0.5,
        filter_thr=0.01,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100
    )
)

{runtime_cfg}
"""


def build_mask2former_cfg(data_root, img_root, work_dir, num_classes, class_names, batch_size, epochs, lr, warmup_iters, num_workers, load_from):
    dataset_cfg = common_dataset_cfg(data_root, img_root, num_classes, class_names, batch_size, num_workers)
    runtime_cfg = common_runtime_cfg(work_dir, epochs, lr, warmup_iters, load_from)

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
