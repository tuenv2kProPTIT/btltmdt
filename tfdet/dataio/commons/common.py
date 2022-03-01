from tfdet.dataio.registry import get_pipeline
from tfdet.utils.shape_utils import shape_list
from tfdet.dataio.registry import register
from dataclasses import dataclass,asdict, field
from typing import Dict, Tuple, Union
from tfdet.dataio.transform import TransformConfig, Transform
import tensorflow as tf
@dataclass
class ComposeConfig(TransformConfig):
    name:str = 'compose'
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
    name:str = 'OneOf'
    last_modified: str = '27/02/2022'
    list_pipeline : Tuple = None 

@register
class OneOf(Transform):
    cfg_class = OneOfConfig
    def __init__(self, cfg: TransformConfig, *args, **kwargs):
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
        idx = len(self.list_functions)
        idx = tf.random.uniform([],0,idx,dtype=tf.int32)
        return self.list_functions[idx](data_dict, training=None)

@dataclass
class KeepAndProcessBBoxConfig(TransformConfig):
    name: str = 'KeepAndProcessBBox'
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