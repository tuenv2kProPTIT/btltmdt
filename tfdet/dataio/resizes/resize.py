from turtle import width
from typing import Dict
import tensorflow as tf 
from dataclasses import dataclass
from tfdet.dataio.resizes import functions as F
from tfdet.dataio.registry import register
from tfdet.utils.serializable import keras_serializable
from tfdet.dataio.transform import TransformConfig,Transform,to_tuple

@dataclass
class LongestMaxSizeConfig(TransformConfig):
    name: str = 'LongestMaxSize'
    last_modified: str = "26/02/2022"
    max_size:int=1024
    interpolation:str='bilinear'
    always_apply:bool=False
    p:float=1.
@dataclass
class RandomScaleConfig(TransformConfig):
    name : str='RandomScale'
    last_modified:str='26/02/2022'
    scale_limit:float=0.1
    interpolation:str='bilinear'
    always_apply:bool=False 
    p:float=0.5
@dataclass
class ResizeConfig(TransformConfig):
    name:str='resize'
    last_modified:str='26/02/2022'
    height:int=None 
    width:int=None 
    interpolation:str='bilinear'
@dataclass
class SmallestMaxSizeConfig(TransformConfig):
    name:str='SmallestMaxSize'
    last_modified:str='26/02/2022'
    max_size:int=1024
    interpolation:str='bilinear'
    always_apply:bool=False
    p:float=1.

@register
@keras_serializable    
class LongestMaxSize(Transform):
    cfg_class = LongestMaxSizeConfig
    def __init__(self, cfg:LongestMaxSizeConfig, *args, **kwargs):
        super().__init__(cfg,*args, **kwargs)
      
    def apply_box(self, bboxes, dict_params=None):
        return bboxes
    def apply_keypoint(self, keypoint, dict_params=None):
        return keypoint
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict:Dict, training=None):
        params = self.get_params(training=training)
        image= F.longest_max_size(
            data_dict['image'],
            max_size=params['max_size'],
            interpolation=params['interpolation']
        )
        data_dict.update(
            image=image
        )
        if 'bboxes' in data_dict:
            bboxes=self.apply_box(data_dict['bboxes'])
            data_dict.update(
                bboxes = bboxes
            )
        if 'keypoint' in data_dict:
            keypoint = self.apply_keypoint(data_dict['keypoint'])
            data_dict.update(
                keypoint = keypoint
            )

        return data_dict

@register
@keras_serializable 
class RandomScale (Transform):
    cfg_class = RandomScaleConfig
    def __init__(self, cfg:RandomScaleConfig, *args, **kwargs):
        super().__init__(cfg,*args, **kwargs)
        self.cfg.scale_limit = to_tuple(self.cfg.scale_limit)
    def get_params(self, training=None):
        param=super().get_params(training=training)
        param.update(scale= tf.random.uniform([],self.cfg.scale_limit[0], self.cfg.scale_limit[1]))
        return param
    def apply_keypoint(self, keypoint, dict_params=None):
        """function is wrong, for future work!
        """
        return keypoint
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict:Dict, training=None):
        scale_param = self.get_params(training=training)
        image= F.scale(
            data_dict['image'],
            scale=scale_param['scale'],
            interpolation=scale_param['interpolation']
        )
        data_dict.update(
            image=image
        )
        if 'keypoint' in data_dict:
            keypoint = self.apply_keypoint(data_dict['keypoint'],scale_param)
            data_dict.update(
                keypoint = keypoint
            )
        return data_dict

@register
@keras_serializable 
class Resize(Transform):
    cfg_class=ResizeConfig
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict:Dict, training=None):
        params=self.get_params()
        image= F.resize(data_dict['image'],height=params['height'],width=params['width'],interpolation=params['interpolation'])
        data_dict.update(
            image=image
        )
        if 'keypoint' in data_dict:
            keypoint = self.apply_keypoint(data_dict['keypoint'],params)
            data_dict.update(
                keypoint = keypoint
            )
        return data_dict

@register
@keras_serializable 
class SmallestMaxSize(Transform):
    cfg_class=SmallestMaxSizeConfig
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict, training=None):
        params = self.get_params(training=training)
        image= F.smallest_max_size(
            data_dict['image'],
            max_size=params['max_size'],
            interpolation=params['interpolation']
        )
        data_dict.update(
            image=image
        )
        if 'bboxes' in data_dict:
            bboxes=self.apply_box(data_dict['bboxes'])
            data_dict.update(
                bboxes = bboxes
            )
        if 'keypoint' in data_dict:
            keypoint = self.apply_keypoint(data_dict['keypoint'])
            data_dict.update(
                keypoint = keypoint
            )

        return data_dict


    


