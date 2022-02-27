from turtle import width
from typing import Dict, Tuple
import tensorflow as tf 
from dataclasses import dataclass,asdict, field
from tfdet.dataio.registry import register
from tfdet.utils.serializable import keras_serializable
from tfdet.dataio.registry import get_pipeline
from tfdet.utils.shape_utils import shape_list
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
    num_parallel_calls=None
    deterministic=None
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
@dataclass
class ComposeConfig(TransformConfig):
    name = 'compose'
    last_modified : str = '27/02/2022'
    list_pipeline : Tuple = None 
@register
class Compose(Transform):
    cfg_class = ComposeConfig
    def __init__(self,cfg:ComposeConfig,*args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
    
        list_pipe =self.cfg.list_pipeline
        list_functions=[]
        list_cfg_pipe = []
        for pipe in list_pipe:
            pipe_instance = get_pipeline(pipe)
            list_functions.append(pipe_instance)
            list_cfg_pipe.append(pipe_instance.get_config())
        self.cfg.list_pipeline =list_cfg_pipe
        self.list_functions = list_functions
    def call(self, data_dict, training=None):
        for pipe in self.list_functions:
            data_dict = pipe(data_dict, training=training)
        return data_dict
@dataclass
class OneOfConfig(TransformConfig):
    name = 'OneOf'
    last_modified: str = '27/02/2022'
    list_pipeline : Tuple = None 

@register
class OneOf(Transform):
    cfg_class = OneOfConfig
    def __init__(self, cfg: TransformConfig, *args, **kwargs):
       
        list_pipe =self.cfg.list_pipeline
        list_functions=[]
        list_cfg_pipe = []
        for pipe in list_pipe:
            pipe_instance = get_pipeline(pipe)
            list_functions.append(pipe_instance)
            list_cfg_pipe.append(pipe_instance.get_config())
        self.cfg.list_pipeline =list_cfg_pipe
        self.list_functions = list_functions
    
    def call(self, data_dict, training=None):
        idx = len(self.list_functions)
        idx = tf.random.uniform([],0,idx,dtype=tf.int32)
        return self.list_functions[idx](data_dict, training=None)

@dataclass
class KeepAndProcessBBoxConfig(TransformConfig):
    name = 'KeepAndProcessBBox'
    keep_dict :Tuple[str] = ('id', 'image', 'bboxes', 'labels', 'mask')
def denormalize_bbox(bbox, rows, cols):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.
    Args:
        bbox (tuple): Normalized bounding box `(y_min,x_min, y_max, x_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
    Raises:
        ValueError: If rows or cols is less or equal zero
    """

    var_multi = tf.stack([rows, cols, rows, cols])
    var_multi  = tf.cast(var_multi, tf.float32)

    var_multi = tf.reshape(var_multi,[1,-1])
    return bbox * var_multi
@register
class KeepAndProcessBBox(Transform):
    cfg_class=KeepAndProcessBBoxConfig
    def apply_box(self, bboxes, dict_params=None):
        return denormalize_bbox(bboxes,dict_params['rows'], dict_params['cols'])
    def call(self, data_dict, training=None):
        params = self.get_params()
        image = data_dict['image']
        h,w = shape_list(image)[-3:-1]
        params['rows']  =  h
        params['cols']  = w
        if 'bboxes' in data_dict:
            bboxes = self.apply_box(data_dict['bboxes'], params)
            data_dict['bboxes'] = bboxes
        keep_dict = params['keep_dict']
        if keep_dict == 'all':
            return data_dict
        return {k:data_dict[k] for k in keep_dict}