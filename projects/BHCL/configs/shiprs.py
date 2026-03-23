# dataset settings
dataset_type = 'ShipRSImageNetDataset'
data_root = '/opt/data/private/data/ShipRSImageNet_V1_DOTA_format/'
backend_args = None

num_classes = 49

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
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline)
)

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

test_dataloader = val_dataloader

leaf_indices = [
    # Level 2
    3,  6,  8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    # Level 3
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48
]

hierarchical_labels = [
    # Level 0
    ('Ship',),
    # Level 1
    ('Ship', 'Warship'),
    ('Ship', 'Merchant'),
    # Level 2
    ('Ship', 'Warship', 'Submarine'),
    ('Ship', 'Warship', 'Aircraft Carrier'),
    ('Ship', 'Warship', 'Destroyer'),
    ('Ship', 'Warship', 'Ticonderoga'),
    ('Ship', 'Warship', 'Frigate'),
    ('Ship', 'Warship', 'Patrol'),
    ('Ship', 'Warship', 'Landing'),
    ('Ship', 'Warship', 'Commander'),
    ('Ship', 'Warship', 'Auxiliary Ship'),
    ('Ship', 'Merchant', 'Container Ship'),
    ('Ship', 'Merchant', 'RoRo'),
    ('Ship', 'Merchant', 'Cargo'),
    ('Ship', 'Merchant', 'Barge'),
    ('Ship', 'Merchant', 'Tugboat'),
    ('Ship', 'Merchant', 'Ferry'),
    ('Ship', 'Merchant', 'Yacht'),
    ('Ship', 'Merchant', 'Sailboat'),
    ('Ship', 'Merchant', 'Fishing Vessel'),
    ('Ship', 'Merchant', 'Oil Tanker'),
    ('Ship', 'Merchant', 'Hovercraft'),
    ('Ship', 'Merchant', 'Motorboat'),
    # Level 3
    ('Ship', 'Warship', 'Aircraft Carrier', 'Enterprise'),
    ('Ship', 'Warship', 'Aircraft Carrier', 'Nimitz'),
    ('Ship', 'Warship', 'Aircraft Carrier', 'Midway'),
    ('Ship', 'Warship', 'Destroyer', 'Atago DD'),
    ('Ship', 'Warship', 'Destroyer', 'Arleigh Burke DD'),
    ('Ship', 'Warship', 'Destroyer', 'Hatsuyuki DD'),
    ('Ship', 'Warship', 'Destroyer', 'Hyuga DD'),
    ('Ship', 'Warship', 'Destroyer', 'Asagiri DD'),
    ('Ship', 'Warship', 'Frigate', 'Perry FF'),
    ('Ship', 'Warship', 'Landing', 'YuTing LL'),
    ('Ship', 'Warship', 'Landing', 'YuDeng LL'),
    ('Ship', 'Warship', 'Landing', 'YuDao LL'),
    ('Ship', 'Warship', 'Landing', 'YuZhao LL'),
    ('Ship', 'Warship', 'Landing', 'Austin LL'),
    ('Ship', 'Warship', 'Landing', 'Osumi LL'),
    ('Ship', 'Warship', 'Landing', 'Wasp LL'),
    ('Ship', 'Warship', 'Landing', 'LSD 41 LL'),
    ('Ship', 'Warship', 'Landing', 'LHA LL'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'Medical Ship'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'Test Ship'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'Training Ship'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'AOE'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'Masyuu AS'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'Sanantonio AS'),
    ('Ship', 'Warship', 'Auxiliary Ship', 'EPF')
]
