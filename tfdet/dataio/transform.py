from turtle import width
from typing import Dict
import tensorflow as tf 
from dataclasses import dataclass
from tfdet.dataio.resizes import functions as F
from tfdet.dataio.registry import register
from tfdet.utils.serializable import keras_serializable
import random
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
    dynamic:bool=True
    always_apply:bool=False
    p:float=1.

class Transform(tf.keras.layers.Layer):
    cfg_class=TransformConfig
    def __init__(self, cfg:TransformConfig, *args, **kwargs):
        if self.cfg.dynamic:
            kwargs['dynamic']=True
        super().__init__(*args, **kwargs)
        self.cfg=cfg 
    def apply_box(self, bboxes, dict_params=None ):
        pass 
    def apply_keypoint(self, keypoint, dict_params=None):
        pass 
    def get_params(self,training=None):
        return self.cfg.as_dict()
    def apply(self, data_dict, training=None):
        pass
    def call(self, data_dict, training=None):
        prob=self.cfg.p
        should_apply_op = tf.cast(
            tf.floor(tf.random_uniform([], dtype=tf.float32) + prob), tf.bool)
        return tf.cond(
            should_apply_op,
            self.apply(data_dict, training=training),
            lambda : data_dict
        )