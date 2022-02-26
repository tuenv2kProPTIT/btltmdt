from dataclasses import dataclass,field
from unicodedata import name
from tfdet.models.necks.registry import register_neck, register
from tfdet.models.config import NeckConfig
from tfdet.models.layers.act_norm_conv import Conv2DNorm
from tfdet.utils.serializable import keras_serializable
import tensorflow as tf
from typing import Dict
@dataclass
class FPNConfig(NeckConfig):
    name = 'FPN'
    last_modified='26/02/2022'
    filters: int = 256
    start_level:int = 0
    end_level: int = -1 
    num_nb_ins: int = 4
    num_nb_outs: int = None 
    
    add_extra_convs:bool=False
    extra_convs_on: str = "on_input"
    relu_before_extra_convs : bool = False
    no_norm_on_lateral=False
    act_layer: str = None
    norm_layer: str =None
    upsample_cfg:Dict=field(default_factory=lambda: dict(interpolation='nearest',size=2))
@register
@keras_serializable
class FPN(tf.keras.Model):
    cfg_class=FPNConfig
    def __init__(self, cfg: FPNConfig, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.cfg = cfg 

        self.lateral_convs = []
        self.fpn_convs = []
        if self.cfg.end_level == -1:
            self.cfg.end_level = self.cfg.num_nb_ins
        for i in range(self.cfg.start_level, self.cfg.end_level):
            l_conv = Conv2DNorm(
                self.cfg.filters,
                1,
                act_cfg=self.cfg.act_layer,
                norm_cfg= self.cfg.norm_layer if not self.cfg.no_norm_on_lateral else None,
            )
            fpn_conv=Conv2DNorm(
                self.cfg.filters,
                3,
                padding='SAME',
                norm_cfg=self.cfg.norm_layer,
                act_cfg=self.cfg.act_layer
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        extra_levels = self.cfg.num_nb_outs - self.cfg.num_nb_ins + self.cfg.start_level
        if self.cfg.add_extra_convs and extra_levels > 0:
            for i in range(extra_levels):
                # if i==0 and self.cfg.extra_convs_on=='on_input':
                extra_fpn_conv  = Conv2DNorm(
                    self.cfg.filters,
                    3,
                    strides=2,
                    padding='SAME',
                    act_cfg=self.cfg.act_layer,
                    norm_cfg= self.cfg.norm_layer,
                ) 
                self.fpn_convs.append(extra_fpn_conv)
        self.fun_upsample = tf.keras.layers.UpSampling2D(**self.cfg.upsample_cfg)
        self.fun_max= tf.keras.layers.MaxPool2D(pool_size=1, strides=2)
    def call(self, inputs, training=False):
        """Forward function."""
        laterals = [
            lateral_conv(inputs[i + self.cfg.start_level], training=training)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i-1] +  self.fun_upsample(laterals[i], training=training)
        outs = [
            self.fpn_convs[i](laterals[i], training=training) for i in range(used_backbone_levels)
        ]

        if self.cfg.num_nb_outs > len(outs):
            if not self.cfg.add_extra_convs:
                for i in range(self.cfg.num_nb_outs - used_backbone_levels):
                    outs.append(self.fun_max(outs[-1]))     
            else:
                if self.cfg.extra_convs_on == 'on_input':
                    extra_source = inputs[self.cfg.num_nb_ins - 1]
                elif self.cfg.extra_convs_on == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.cfg.extra_convs_on == 'on_output':
                    extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.cfg.num_nb_outs):
                    if self.cfg.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](tf.nn.relu(outs[-1]), training=training) )
                    else:
                        outs.append(self.fpn_convs[i](outs[-1], training=training))
        return outs 



