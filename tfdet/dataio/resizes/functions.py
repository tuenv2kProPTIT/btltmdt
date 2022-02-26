import tensorflow as tf 

from tfdet.utils.shape_utils import shape_list
from typing import Any, Callable, Dict, List, Sequence, Tuple
def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))

def _func_max_size(img, max_size, interpolation, func):
    height, width =  shape_list(img)[-3:-1]
    scale = max_size / float(func(width, height))
    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img
def smallest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, min)

def longest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, max)


def resize(img, height, width, interpolation=tf.image.ResizeMethod.BILINEAR):
    img_height, img_width =  shape_list(img)[-3:-1]
    cond = tf.logical_and(tf.equal(height, img_height), tf.equal(width, img_width))
    return tf.cond(cond, lambda : img, lambda :tf.image.resize(img, [height, width], method=interpolation))
    


def scale(img, scale, interpolation=tf.image.ResizeMethod.BILINEAR):
    height, width = shape_list(img)[-3:-1]
    new_height, new_width =tf.cast( tf.cast(height,tf.float32) * scale, tf.int32), tf.cast( tf.cast(width,tf.float32) * scale, tf.int32)
    return resize(img, new_height, new_width, interpolation)