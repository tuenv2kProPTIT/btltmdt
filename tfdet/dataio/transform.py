from turtle import width
from typing import Dict, Tuple, Union
import tensorflow as tf 
from dataclasses import dataclass,asdict, field
from tfdet.dataio.registry import register
from tfdet.utils.serializable import keras_serializable
def to_tuple(x):
    if isinstance(x,list):
        assert len(x) == 2
        return tuple(x)
    if isinstance(x,tuple):
        assert len(x)==2
        return x 
    return (x,x)
@dataclass
class TransformConfig:
    name='transform'
    last_modified: str='26/02/2022'

    dynamic:bool=False
    
    always_apply:bool=False
    p:float=1.

    # option map ds 
    num_parallel_calls:Union[int,str]=None
    deterministic:Union[int,str]=None

class Transform(tf.keras.layers.Layer):
    cfg_class=TransformConfig
    def __init__(self, cfg:TransformConfig, *args, **kwargs):
        if cfg.dynamic:
            kwargs['dynamic']=True
        if cfg.num_parallel_calls and cfg.num_parallel_calls == 'AUTO':
            cfg.num_parallel_calls = tf.data.AUTOTUNE
        if cfg.deterministic and cfg.deterministic == 'true':
            cfg.deterministic = True 
        if cfg.deterministic and cfg.deterministic == 'false':
            cfg.deterministic = False
        super().__init__(*args, **kwargs)
        self.cfg=cfg 
    def apply_box(self, bboxes, dict_params=None ):
        pass 
    def apply_keypoint(self, keypoint, dict_params=None):
        pass 
    def get_params(self,training=None):
        return asdict(self.cfg)
    def apply(self, data_dict, training=None):
        pass
    def call(self, data_dict, training=None):
        prob=self.cfg.p
        should_apply_op = tf.cast(
            tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
        return tf.cond(
            should_apply_op,
            lambda : self.apply({k:tf.identity(v) for k,v in data_dict.items()}, training=training),
            lambda : data_dict
        )
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            asdict(self.cfg)
        )
        return cfg
