from statistics import variance
from numpy import var
import tensorflow as tf 
from tfdet.utils.shape_utils import shape_list
from typing import Tuple

def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def center_crop(img: tf.Tensor, crop_height: int, crop_width: int):
    height, width =shape_list(img)[-3:-1]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[...,y1:y2, x1:x2,:]
    return img

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
    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")
    shape = shape_list(bbox)[-1]
    if shape ==4:
        var_multi = tf.constant([rows*1., cols*1., rows*1., cols*1.], dtype=tf.float32)
    else:
        var_multi  = tf.constant([rows*1., cols*1., rows*1., cols*1.,] + [1.,] *int(shape - 4) , dtype=tf.float32)
    var_multi = tf.reshape(var_multi,[1,-1])
    return bbox * var_multi
def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    Args:
        bbox (tuple): Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: Normalized bounding box `(x_min, y_min, x_max, y_max)`.
    Raises:
        ValueError: If rows or cols is less or equal zero
    """
    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")
    shape = shape_list(bbox)[-1]
    if shape ==4:
        var_multi = tf.constant([rows*1., cols*1., rows*1., cols*1.], dtype=tf.float32)
    else:
        var_multi  = tf.constant([rows*1., cols*1., rows*1., cols*1.] + [1.,] *int(shape - 4) , dtype=tf.float32)
    var_multi = tf.reshape(var_multi,[1,-1])
    return tf.math.divide_no_nan(bbox,  var_multi) 

def crop_bbox_by_coords(
    bbox, crop_coords: Tuple[int, int, int, int], crop_height: int, crop_width: int, rows: int, cols: int
):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.
    Args:
        bbox (tuple): A cropped box `(x_min, y_min, x_max, y_max)`.
        crop_coords (tuple): Crop coordinates `(x1, y1, x2, y2)`.
        crop_height (int):
        crop_width (int):
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    
    shape = shape_list(bbox)[-1]
    x1, y1, _, _ = crop_coords
    crop_cord =[y1, x1, y1, x1]
    if shape > 4:
        crop_cord.extend([0. for i in range(shape-4)])
    crop_coords =tf.constant(crop_cord,dtype=tf.float32)
    cropped_bbox = bbox - crop_cord
    return normalize_bbox(cropped_bbox, crop_height, crop_width)

def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def bbox_center_crop(bbox, crop_height: int, crop_width: int, rows: int, cols: int):
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)

def bbox_random_crop(
    bbox:tf.Tensor, crop_height: int, crop_width: int, h_start: float, w_start: float, rows: int, cols: int
):
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)