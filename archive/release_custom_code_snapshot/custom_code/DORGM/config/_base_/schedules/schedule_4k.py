# optimizer
# optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001, nesterov=False)
optimizer = dict(type='Adam', lr=0.0003)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=2000,
        by_epoch=False)
]
# training schedule for 4k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1500, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20, max_keep_ckpts=2, save_best='mIoU', rule='greater', published_keys=['state_dict'], save_begin=500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=500))
