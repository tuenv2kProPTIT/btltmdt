from dataclasses import dataclass
from tfdet.utils.constants import IMAGE_KEY, BBOX_KEY, LABEL_KEY
import tensorflow as tf
import PIL.Image
import hashlib
import io,json,os
from tqdm import tqdm
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
def serializable_feature(image_dict):
    image_path = image_dict['image_path']
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size
    key = hashlib.sha256(encoded_jpg).hexdigest()
    feature_dict = {
      'image/height':
          _int64_feature(height),
      'image/width':
          _int64_feature(width),
      'image/filename':
         _bytes_feature(image_path.encode('utf8')),
      'image/key/sha256':
         _bytes_feature(key.encode('utf8')),
      'image/encoded':
         _bytes_feature(encoded_jpg),
      'image/format':
         _bytes_feature('jpeg'.encode('utf8')),
    }
    num_annotations_skipped =0
    bbox_annotations=image_dict['bboxes']
    label_annotation = image_dict['labels']
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []

    for bbox,label in zip(bbox_annotations, label_annotation):
        y_min,x_min,y_max,x_max = bbox 
        xmin.append(float(x_min) / width)
        xmax.append(float(x_max) / width)
        ymin.append(float(y_min) / height)
        ymax.append(float(y_max) / height)
        category_ids.append(label['id'])
        category_names.append(str(label['name']).encode("utf8"))
    feature_dict.update({
        'image/object/bbox/xmin':
            float_list_feature(xmin),
        'image/object/bbox/xmax':
            float_list_feature(xmax),
        'image/object/bbox/ymin':
           float_list_feature(ymin),
        'image/object/bbox/ymax':
           float_list_feature(ymax),
        'image/object/class/text':
            bytes_list_feature(category_names),
        'image/object/class/label':
            int64_list_feature(category_ids)
    })
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example 
def convert_dataset_to_tfrecord(
    datasets,
    num_shards=5,
    output_dir='./datasets',
    total_log=200
):
    '''datasets = list[imge_dict]
    image_dict:
        +> image_path,
        +> bboxes
        +> labels
    '''
    os.makedirs(output_dir,exist_ok=True)
    writers = [
      tf.io.TFRecordWriter(os.path.join(output_dir , '%06d-of-%06d.tfrecord' %
                           (i, num_shards))) for i in range(num_shards)
    ]

    for idx,image_object in tqdm(enumerate(datasets)):
        example =serializable_feature(image_object)
        writers[idx % num_shards].write(example.SerializeToString())
        if idx % total_log == 0:
            print(f"working on {idx}")
    for writer in writers:
        writer.close()

