from dataclasses import dataclass
import tensorflow as tf

from tfdet.utils.constants import BBOX_KEY
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


_CONFIG_KEY={
    'image/height':{'key_to_features': tf.io.FixedLenFeature((), tf.int64, -1), 'key_encode':_int64_feature},
    'image/width':{'key_to_features': tf.io.FixedLenFeature((), tf.int64, -1), 'key_encode':_int64_feature},
    'image/filename':{'key_to_features':tf.io.FixedLenFeature((), tf.string), 'key_encode':_bytes_feature},
    'image/encoded':{ 'key_to_features':tf.io.FixedLenFeature((), tf.string), 'key_encode':_bytes_feature},
    'image/object/bbox/xmin':{'key_to_features':tf.io.VarLenFeature(tf.float32), 'key_encode':float_list_feature},
    "image/object/bbox/xmax":{'key_to_features':tf.io.VarLenFeature(tf.float32), 'key_encode':float_list_feature},
    "image/object/bbox/ymin":{'key_to_features':tf.io.VarLenFeature(tf.float32), 'key_encode':float_list_feature},
    "image/object/bbox/ymax":{'key_to_features':tf.io.VarLenFeature(tf.float32), 'key_encode':float_list_feature},
    "image/object/class/text":{'key_to_features':tf.io.FixedLenFeature((), tf.string), 'key_encode':bytes_list_feature},
    "image/object/class/label":{'key_to_features':tf.io.FixedLenFeature((), tf.string), 'key_encode':bytes_list_feature}
}

class Constant:
    HEIGHT='image/height'
    WIDTH= 'image/width'
    FILE_NAME='image/filename'
    IMAGE_ENCODE='image/encoded'
    XMIN='image/object/bbox/xmin'
    XMAX='image/object/bbox/xmax'
    YMIN='image/object/bbox/ymin'
    YMAX='image/object/bbox/ymax'
    CLASS_NAME='image/object/class/text'
    CLASS_ID='image/object/class/label'

class TfExampleDecoder:
    def __init__(self, list_key):
        self._keys_to_features = {
            _CONFIG_KEY[k]['key_to_features'] for k in list_key
        }
    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        return tf.io.decode_image(parsed_tensors[Constant.IMAGE_ENCODE], channels=3, expand_animations=False)

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors[Constant.XMIN]
        xmax = parsed_tensors[Constant.XMAX]
        ymin = parsed_tensors[Constant.YMIN]
        ymax = parsed_tensors[Constant.YMAX]
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors[Constant.XMIN]
        xmax = parsed_tensors[Constant.XMAX]
        ymin = parsed_tensors[Constant.YMIN]
        ymax = parsed_tensors[Constant.YMAX]
        return(xmax - xmin) * (ymax - ymin)
    def decode(self, serialized_example):
        """Decode the serialized example.
        Args:
        serialized_example: a single serialized tf.Example string.
        Returns:
        decoded_tensors: a dictionary of tensors with the following fields:
            - image: a uint8 tensor of shape [None, None, 3].
            - source_id: a string scalar tensor.
            - height: an integer scalar tensor.
            - width: an integer scalar tensor.
            - groundtruth_classes: a int64 tensor of shape [None].
            - groundtruth_is_crowd: a bool tensor of shape [None].
            - groundtruth_area: a float32 tensor of shape [None].
            - groundtruth_boxes: a float32 tensor of shape [None, 4].
            - groundtruth_instance_masks: a float32 tensor of shape
                [None, None, None].
            - groundtruth_instance_masks_png: a string tensor of shape [None].
        """
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)

        


@dataclass
class ConfigDataSet:
    name : str = 'ObjectDetectionDataSetSubSetname'
    dir : str = 'Dir of datasets'
    pandas_file: str= 'pandas file path save object'


