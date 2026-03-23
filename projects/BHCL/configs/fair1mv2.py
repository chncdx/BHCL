# dataset settings
dataset_type = 'FAIR1Mv2Dataset'
data_root = '/opt/data/private/data/FAIR1M_V2_Cropped_DOTA_format/'
backend_args = None

num_classes = 39

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.RandomShift',
        prob=1,
        max_shift_px=32),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/labelTxt/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

# The validation set is used during training and its evaluation results should not
# be included in the publication. It serves only for monitoring the training process.
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))

leaf_indices = [
    # Level 1
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
]

hierarchical_labels = [
    # Level 0
    ('Ship',),
    ('Vehicle',),
    ('Airplane',),
    ('Court',),
    ('Road',),
    # Level 1
    ('Ship', 'Passenger Ship'),
    ('Ship', 'Motorboat'),
    ('Ship', 'Fishing Boat'),
    ('Ship', 'Tugboat'),
    ('Ship', 'Engineering Ship'),
    ('Ship', 'Liquid Cargo Ship'),
    ('Ship', 'Dry Cargo Ship'),
    ('Ship', 'Warship'),
    ('Vehicle', 'Small Car'),
    ('Vehicle', 'Bus'),
    ('Vehicle', 'Cargo Truck'),
    ('Vehicle', 'Dump Truck'),
    ('Vehicle', 'Van'),
    ('Vehicle', 'Trailer'),
    ('Vehicle', 'Tractor'),
    ('Vehicle', 'Excavator'),
    ('Vehicle', 'Truck Tractor'),
    ('Airplane', 'Boeing737'),
    ('Airplane', 'Boeing747'),
    ('Airplane', 'Boeing777'),
    ('Airplane', 'Boeing787'),
    ('Airplane', 'ARJ21'),
    ('Airplane', 'A220'),
    ('Airplane', 'A321'),
    ('Airplane', 'A330'),
    ('Airplane', 'A350'),
    ('Airplane', 'C919'),
    ('Court', 'Baseball Field'),
    ('Court', 'Basketball Court'),
    ('Court', 'Football Field'),
    ('Court', 'Tennis Court'),
    ('Road', 'Roundabout'),
    ('Road', 'Intersection'),
    ('Road', 'Bridge')
]
