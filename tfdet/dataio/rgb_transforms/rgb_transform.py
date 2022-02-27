from statistics import mean
from tfdet.dataio.registry import register
from tfdet.dataio.transform import asdict,keras_serializable,dataclass,Transform,TransformConfig
from typing import Dict, Tuple
from tfdet.dataio import functional as F
import tensorflow as tf 
from tfdet.utils.shape_utils import shape_list
@dataclass
class NormalizeConfig(TransformConfig):
    name : str  = 'Normalize'
    last_modified: str = '27/02/2022'
    max_pixel_value: float = 255.
    mean: Tuple[float,float,float] = (0.485, 0.456, 0.406)
    std : Tuple[float, float, float] = (0.229, 0.224, 0.225)
    p: float = 1.
@register
@keras_serializable
class Normalize(Transform):
    cfg_class = NormalizeConfig
    def __init__(self, cfg: TransformConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg.mean = (self.cfg.mean, self.cfg.mean, self.cfg.mean) if isinstance(self.cfg.mean, float) else self.cfg.mean
        self.cfg.std = (self.cfg.std, self.cfg.std, self.cfg.std) if  isinstance(self.cfg.std, float) else self.cfg.std
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict, training=None):
        params = self.get_params()
        mean = tf.constant(params['mean'],tf.float32)
        std = tf.constant(params['std'], tf.float32)
        image = data_dict['image']
        shape= shape_list(image)
        if len(shape) == 4:
            mean = tf.expand_dims(mean,0)
            std = tf.expand_dims(std, 0)
        image = F.normalize(
            data_dict['image'],
            mean=mean,
            std=std,
            max_pixel_value=params['max_pixel_value']
        )
        data_dict.update(image=image)
        return data_dict
