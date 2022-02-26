from dataclasses import Field, dataclass, field
from random import shuffle
from typing import Dict
import tensorflow as tf 

from tfdet.dataio.registry import register


class TfExampleDecoder:
    def __init__(self, include_mask=False, regenerate_source_id=False):
        self._include_mask = include_mask
        self._regenerate_source_id = regenerate_source_id
        self._keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/height': tf.io.FixedLenFeature((), tf.int64, -1),
            'image/width': tf.io.FixedLenFeature((), tf.int64, -1),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/filename': tf.io.FixedLenFeature((), tf.string) 

        }
        if include_mask:
            self._keys_to_features.update({
                'image/object/mask':
                    tf.io.VarLenFeature(tf.string),
            })

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        return tf.io.decode_image(parsed_tensors['image/encoded'], channels=3, expand_animations=False)

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""
        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors['image/height']
        width = parsed_tensors['image/width']
        masks = parsed_tensors['image/object/mask']
        return tf.cond(
            tf.greater(tf.shape(masks)[0], 0),
            lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            lambda: tf.zeros([0, height, width], dtype=tf.float32))

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
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

        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors['image/height'], -1),
            tf.equal(parsed_tensors['image/width'], -1))
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

        parsed_tensors['image/height'] = tf.where(decode_image_shape,
                                                image_shape[0],
                                                parsed_tensors['image/height'])
        parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1],
                                                parsed_tensors['image/width'])

        if self._include_mask:
            masks = self._decode_masks(parsed_tensors)

        decoded_tensors = {
            "filename":parsed_tensors['image/filename'],
            'image': image,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_area': areas,
            'groundtruth_boxes': boxes,
        }
        if self._include_mask:
            decoded_tensors.update({
                'groundtruth_instance_masks': masks,
                'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
            })
        return decoded_tensors

@dataclass
class InputProcessingConfig:
    name : str = 'InputReadRecordFiles'
    last_modified: str  = '26/02/2022'
    pattern_file: str = None 
    seed : int = 2022
    shuffle: bool = False 
    bucket_size_shuffle: int =64
    debug: bool = False
    include_mask:bool=False
class InputProcessing:
    cfg_class = InputProcessingConfig
    def __init__(self, cfg:InputProcessingConfig, *args, **kwargs):
        self.cfg =cfg 
    def __call__(self):
        dataset = tf.data.Dataset.list_files(
            self.cfg.pattern_file,
            shuffle = self.cfg.shuffle,
            seed=self.cfg.seed
        )
        def _prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(filename).prefetch(1)
            return dataset
        dataset = dataset.interleave(_prefetch_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.deterministic = self.cfg.debug 
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True

        dataset = dataset.with_options(options)

        if self.cfg.shuffle:
            dataset = dataset.shuffle(self.cfg.bucket_size_shuffle, seed=self.cfg.seed)
        example_decoder = TfExampleDecoder(
            include_mask=self.cfg.include_mask
        )
        map_fn = lambda  value: self.dataset_parser(value, example_decoder)
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    @tf.autograph.experimental.do_not_convert
    def dataset_parser(self, value, example_decoder):
        with tf.name_scope('parser'):
            data = example_decoder.decode(value)
            image = data['image']
            boxes = data['groundtruth_boxes']
            classes = data['groundtruth_classes']
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
            areas = data['groundtruth_area']
            image_masks = data.get('groundtruth_instance_masks', [])
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
            mask = tf.ones_like(classes)
        data_parser= {
            'filename':data['filename'],
            'image':image,
            'bboxes':boxes,
            "labels":classes,
            'areas':areas,
            "mask":mask
        }
        if self.cfg.include_mask:
            data_parser.update(
                image_masks=image_masks
            )
        return data_parser
