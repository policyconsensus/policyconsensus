import hydra

def create_rgb_encoder(to_use, model_library):
    assert to_use in [
        'DpRgbEncoder', 
        'RobomimicRgbEncoder'
    ]
    return hydra.utils.instantiate(model_library[to_use])


def create_pointcloud_encoder(to_use, model_library):
    assert to_use in [
        'Dp3PointcloudEncoder', 
        'PointTransformerEncoder', 
        'ParticleGraphEncoder',
    ]
    return hydra.utils.instantiate(model_library[to_use])

def create_dino_pointcloud_encoder(to_use, model_library):
    assert to_use in [
        'DinoDp3PointcloudEncoder', 
    ]
    return hydra.utils.instantiate(model_library[to_use])

def create_depth_encoder(to_use, model_library):
    assert to_use in [
        'RobomimicDepthEncoder',
    ]
    return hydra.utils.instantiate(model_library[to_use])

def create_tactile_encoder(to_use, model_library):
    assert to_use in [
        'RobomimicTactileEncoder',
    ]
    return hydra.utils.instantiate(model_library[to_use])