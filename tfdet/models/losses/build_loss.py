

from tfdet.models.losses.focal_loss import FocalLoss,FocalLossConfig
from tfdet.models.losses.smooth_l1_loss import SmoothL1Loss, SmoothL1LossConfig

def build_loss(cfg):
    if cfg.get("name",None) == 'FocalLoss'.lower():
        return  FocalLoss(FocalLossConfig(**cfg)) 
    if cfg.get("name",None) == 'SmoothL1Loss'.lower():
        return SmoothL1Loss(SmoothL1LossConfig(**cfg)) 
        



