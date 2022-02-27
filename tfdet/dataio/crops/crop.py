from turtle import width
from tfdet.dataio.transform import TransformConfig, Transform
from tfdet.dataio.registry import register
from dataclasses import dataclass,field
from typing import Dict,Tuple,Sequence

from tfdet.utils.serializable import keras_serializable
from tfdet.dataio.crops import functions as F
from tfdet.dataio.resizes import functions as  FR
import tensorflow as tf
import math
from tfdet.utils.shape_utils import shape_list 
@dataclass
class CenterCropConfig(TransformConfig):
    name : str='CenterCrop'
    last_modified: str = '26/02/2022'
    height: int = None 
    width: int = None 

@register
@keras_serializable 
class CenterCrop(Transform):
    cfg_class=CenterCropConfig
    def apply_box(self, bboxes, dict_params=None):
        return F.bbox_center_crop(
            bboxes,
            dict_params['height'],
            dict_params['width'],
            dict_params['rows'],
            dict_params['cols']
        )
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict, training=None):
        params=self.get_params(training=training)
        shape = shape_list(data_dict['image'])[-3:-1]
        params['rows'] = shape[0]
        params['cols'] = shape[1]
        image = F.center_crop(
            data_dict['image'],
            crop_height=params['height'],
            crop_width=params['width'],

        )
        data_dict.upadte(
            image=image 
        )

        if 'bboxes' in data_dict:
            bboxes = self.apply_box(data_dict['bboxes'], params)
            data_dict.update(
                bboxes=bboxes
            )

        return data_dict


@dataclass
class BaseRandomSizedCropConfig(TransformConfig):
    name : str = "BaseRandomSizedCrop"
    last_modified: str="26/02/2022"
    height : int = None 
    width  : int = None 
    interpolation: str = "bilinear"

class _BaseRandomSizedCrop(Transform):
    # Base class for RandomSizedCrop and RandomResizedCrop
    cfg_class=BaseRandomSizedCropConfig
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict, training=None):
        img = data_dict['image']
        params = self.get_params(training=training)
        shape = shape_list(img)[-3:-1]
        params['rows'] = shape[0]
        params['cols'] = shape[1]
        crop = F.random_crop(img, params['crop_height'], params['crop_width'],params['h_start'], params['w_start'])
        img= FR.resize(crop, self.height, self.width, params['interpolation'])
        data_dict.update(
            image=img 
        )
        if 'bboxes' in data_dict:
            bboxes = self.apply_box(data_dict['bboxes'], params)
            data_dict.update(
                bboxes=bboxes
            )
        return data_dict
    
    def apply_box(self, bbox, dict_params):
        return F.bbox_random_crop(bbox, dict_params['crop_height'], dict_params['crop_width'], dict_params['h_start'], 
                                dict_params['w_start'], dict_params['rows'], dict_params['cols'])


@dataclass
class RandomSizedCropConfig(BaseRandomSizedCropConfig):
    """Crop a random part of the input and rescale it to some size.
    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1."""
    
    name='RandomSized'
    last_modified: str = '26/02/2022'
    min_max_height : Tuple[int,int] =None
    w2h_ratio : float = 1.0 
@register
@keras_serializable 
class RandomSizedCrop(_BaseRandomSizedCrop):

    cfg_class=RandomSizedCropConfig

    def get_params(self, training=None):
        params=super().get_params(training)
        crop_height =tf.random.uniform([],params['min_max_height'][0], params['min_max_height'][1], dtype=tf.int32)
        params.update(
            crop_height=crop_height,
            crop_width= tf.cast(crop_height * params['w2h_ratio'], tf.int32),
            h_start = tf.random.uniform([]),
            w_start=tf.random.uniform([])
        )
        return params

@dataclass
class RandomResizedCropConfig(BaseRandomSizedCropConfig):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.
    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1."""
    
    name: str = 'RandomResizedCrop'
    last_modified: str = '26/02/2022'
    scale: Tuple[float,float]=field(default_factory= lambda: (0.08, 1.0))
    ratio: Tuple[float,float]=field(default_factory= lambda: (0.75, 1.3333333333333333) )
    p: float = 1.0
@register
@keras_serializable 
class RandomResizedCrop(_BaseRandomSizedCrop):
    cfg_class = RandomResizedCropConfig
    @tf.function(experimental_relax_shapes=True)
    def apply(self, data_dict, training=None):
        img = data_dict['image']
        params = self.get_params(training=training)
        shape = shape_list(img)[-3:-1]
        params['rows'] = shape[0]
        params['cols'] = shape[1]
        area = tf.cast(shape[0] * shape[1], tf.float32) 


        def condition(index, w_pred, h_pred):
            out_of_index =tf.less(index, 10)
            out_of_width =tf.logical_and(tf.less_equal(w_pred, shape[1]), tf.greater(w_pred, 0) )
            out_of_height = tf.logical_and(tf.less_equal(h_pred, shape[1]), tf.greater(h_pred, 0) )
            return tf.logical_and(tf.logical_and(out_of_height, out_of_width), out_of_index)
        
        def body(index, w_pred, h_pred):
            target_area = tf.random.uniform([],params['scale'][0], params['scale'][1], dtype=tf.float32)  * area
            log_ratio = (tf.math.log(params['ratio'][0]), tf.math.log(params['ratio'][1]))
            aspect_ratio = tf.exp(tf.random.uniform([], log_ratio[0], log_ratio[1], dtype=tf.float32) )
            
            w = tf.cast(tf.round(tf.sqrt(  target_area * aspect_ratio )), tf.int32)
            h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio  )), tf.int32)

            return index + 1, w, h
        # swap_memory
        index = tf.constant(0, dtype=tf.int32)
        w = tf.constant(0, dtype=tf.int32)
        h = tf.constant(0, dtype=tf.int32)
        index, w, h = tf.while_loop(condition, body, loop_vars=[index, w, h], swap_memory=False)
        out_of_width = tf.logical_and(tf.less_equal(w, shape[1]), tf.greater(w, 0) )
        out_of_height = tf.logical_and(tf.less_equal(h, shape[1]), tf.greater(h, 0) )

        def inside_attemp(w,h):
            i=tf.random.uniform([],0, shape[0] - h, dtype=tf.int32)
            j = tf.random.uniform([], 0 , shape[1] - w, dtype=tf.int32)

            return {
                "crop_height": tf.cast(h,tf.int32),
                "crop_width":  tf.cast(w,tf.int32),
                "h_start": tf.cast(i,tf.float32)  /(tf.cast(shape[0] - h, tf.float32) + 1e-10),
                "w_start": tf.cast(j,tf.float32)  / (tf.cast(shape[1] - w,tf.float32) + 1e-10),
            }

        def outside_attemp():
            in_ratio = shape[1] / shape[0]
            if in_ratio < min(params['ratio']):
                w = shape[1]
                h = tf.cast(tf.round(tf.cast(w,tf.float32) / min(params['ratio'])), tf.int32)
            elif in_ratio > max(params['ratio']):
                h = shape[0]
                w = tf.cast(tf.round(tf.cast(h,tf.float32) * max(params['ratio'])), tf.int32)
            else:  # whole image
                w = shape[1]
                h = shape[0]
            i = tf.round(tf.cast(shape[0] - h,tf.float32) / 2.)
            j = tf.round(tf.cast(shape[1] - w,tf.float32) / 2.)
            
            return {
                "crop_height": tf.cast(h,tf.int32),
                "crop_width":  tf.cast(w,tf.int32),
                "h_start": i  /(tf.cast(shape[0] - h, tf.float32) + 1e-10),
                "w_start": j  / (tf.cast(shape[1] - w,tf.float32) + 1e-10),
            }
        encode = tf.cond(tf.logical_and(out_of_width, out_of_height),lambda : inside_attemp(w,h),outside_attemp)
        params.update(encode)
        crop = F.random_crop(img, params['crop_height'], params['crop_width'],params['h_start'], params['w_start'])
        img= FR.resize(crop, params['height'], params['width'], params['interpolation'])
        data_dict.update(
            image=img 
        )
        # print(params)
        if 'bboxes' in data_dict:
            bboxes = self.apply_box(data_dict['bboxes'], params)
            data_dict.update(
                bboxes=bboxes
            )
        return data_dict

      
