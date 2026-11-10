weight = 'None'
resume = False
evaluate = True
test_only = False
seed = 9173
save_path = 'runs/ptv3/seed9173'
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
param_dicts = [dict(keyword='block', lr=0.0001)]
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
        use_color=False,
        loop=20),
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
    type='DefaultSegmentorV2',
    num_classes=6,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(512, 512, 512, 512, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(512, 512, 512, 512),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('Private', )),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
