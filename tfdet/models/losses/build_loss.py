

from tfdet.models.losses.focal_loss import FocalLoss,FocalLossConfig
from tfdet.models.losses.smooth_l1_loss import SmoothL1Loss, SmoothL1LossConfig
_loss_register={}
def register(cls):
    cfg=cls.cfg_class 
    name=cfg.name.lower()
    _loss_register[name]={
        'config':cfg,
        'instance':cls
    }

def build_loss(cfg):
    name = cfg.get("name",None)
    if name:
        name=name.lower()
    if name in _loss_register:
        return _loss_register[name]['instance'](_loss_register[name]['config'](**cfg))
    else:
        raise Exception(f"class {name} didn't register at anywhere")
        



