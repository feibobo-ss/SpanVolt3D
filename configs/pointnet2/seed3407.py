weight = None
resume = False
evaluate = True
test_only = False
seed = 42
save_path = 'runs/pointnet2/seed3407'
num_worker = 2
batch_size = 1
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 200
eval_epoch = 10
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = True
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.0
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
dataset_type = 'PrivateDataset'
data_root = 'data/las_vegetation6_final_v1'
grid_size = 0.08
point_max = 60000
class_names = [
    'tower', 'conductor', 'jumper_conductor', 'insulator', 'shield_wire',
    'vegetation'
]
data = dict(
    num_classes=6,
    ignore_index=-1,
    names=[
        'tower', 'conductor', 'jumper_conductor', 'insulator', 'shield_wire',
        'vegetation'
    ],
    train=dict(
        type='PrivateDataset',
        split='train',
        data_root='data/las_vegetation6_final_v1',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.05,
                dropout_application_ratio=0.15),
            dict(
                type='RandomRotate',
                angle=[-0.1, 0.1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.95, 1.05]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.08,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=60000, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        use_color=False),
    val=dict(
        type='PrivateDataset',
        split='val',
        data_root='data/las_vegetation6_final_v1',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.08,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=60000, mode='center'),
            dict(type='CenterShift', apply_z=False),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        use_color=False),
    test=dict(
        type='PrivateDataset',
        split='test',
        data_root='data/las_vegetation6_final_v1',
        transform=[dict(type='CenterShift', apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=None,
            crop=dict(type='SphereCrop', point_max=60000, mode='all'),
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='GridCoord', grid_size=0.08),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', ))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
            }]]),
        use_color=False))
save_test_predictions = False
model = dict(
    type='DefaultSegmentor',
    backbone=dict(type='PointTransformer-Seg50', in_channels=3, num_classes=6),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(type='MultiStepLR', milestones=[0.6, 0.8], gamma=0.1)
