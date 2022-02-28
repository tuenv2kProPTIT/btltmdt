import functools
from typing import List, Tuple

from absl import logging
import tensorflow as tf 
from tfdet.utils.shape_utils import shape_list
T=tf.Tensor

def batch_map_fn(map_fn, inputs, *args):
  """Apply map_fn at batch dimension."""
  if isinstance(inputs[0], (list, tuple)):
    batch_size = len(inputs[0])
  else:
    batch_size = inputs[0].shape.as_list()[0]

  if not batch_size:
    # handle dynamic batch size: tf.vectorized_map is faster than tf.map_fn.
    return tf.vectorized_map(map_fn, inputs, *args)

  outputs = []
  for i in range(batch_size):
    outputs.append(map_fn([x[i] for x in inputs]))
  return [tf.stack(y) for y in zip(*outputs)]

  
def nms(params, boxes: T, scores: T, classes: T,
        padded: bool) -> Tuple[T, T, T, T]:
  """Non-maximum suppression.
  Args:
    params: a dict of parameters.
    boxes: a tensor with shape [N, 4], where N is the number of boxes. Box
      format is [y_min, x_min, y_max, x_max].
    scores: a tensor with shape [N].
    classes: a tensor with shape [N].
    padded: a bool vallue indicating whether the results are padded.
  Returns:
    A tuple (boxes, scores, classes, valid_lens), where valid_lens is a scalar
    denoting the valid length of boxes/scores/classes outputs.
  """
  nms_configs = params['nms_configs']
  method = nms_configs['method']
  max_output_size = nms_configs['max_output_size']

  if method == 'hard' or not method:
    # hard nms.
    sigma = 0.0
    iou_thresh = nms_configs['iou_thresh'] or 0.5
    score_thresh = nms_configs['score_thresh'] or float('-inf')
  elif method == 'gaussian':
    sigma = nms_configs['sigma'] or 0.5
    iou_thresh = 1.0
    score_thresh = nms_configs['score_thresh'] or 0.001
  else:
    raise ValueError('Inference has invalid nms method {}'.format(method))

  # TF API's sigma is twice as the paper's value, so here we divide it by 2:
  # https://github.com/tensorflow/tensorflow/issues/40253.
  nms_top_idx, nms_scores, nms_valid_lens = tf.raw_ops.NonMaxSuppressionV5(
      boxes=boxes,
      scores=scores,
      max_output_size=max_output_size,
      iou_threshold=iou_thresh,
      score_threshold=score_thresh,
      soft_nms_sigma=(sigma / 2),
      pad_to_max_output_size=padded)

  nms_boxes = tf.gather(boxes, nms_top_idx)
  nms_classes = tf.cast(
      tf.gather(classes, nms_top_idx), boxes.dtype)
  return nms_boxes, nms_scores, nms_classes, nms_valid_lens


def topk_class_boxes(params, cls_outputs: T,
                     box_outputs: T) -> Tuple[T, T, T, T]:
  """Pick the topk class and box outputs."""
  batch_size = shape_list(cls_outputs)[0]
  num_classes = params['num_classes']
  max_nms_inputs = params['nms_configs'].get('max_nms_inputs', 0)
  if max_nms_inputs > 0:
    # Prune anchors and detections to only keep max_nms_inputs.
    # Due to some issues, top_k is currently slow in graph model.
    logging.info('use max_nms_inputs for pre-nms topk.')
    cls_outputs_reshape = tf.reshape(cls_outputs, [batch_size, -1])
    _, cls_topk_indices = tf.math.top_k(
        cls_outputs_reshape, k=max_nms_inputs, sorted=False)
    indices = cls_topk_indices // num_classes
    classes = cls_topk_indices % num_classes
    cls_indices = tf.stack([indices, classes], axis=2)

    cls_outputs_topk = tf.gather_nd(cls_outputs, cls_indices, batch_dims=1)
    box_outputs_topk = tf.gather_nd(
        box_outputs, tf.expand_dims(indices, 2), batch_dims=1)
  else:
    logging.info('use max_reduce for pre-nms topk.')
    # Keep all anchors, but for each anchor, just keep the max probablity for
    # each class.
    cls_outputs_idx = tf.math.argmax(cls_outputs, axis=-1, output_type=tf.int32)
    num_anchors = shape_list(cls_outputs)[1]
    classes = cls_outputs_idx
    indices = tf.tile(
        tf.expand_dims(tf.range(num_anchors), axis=0), [batch_size, 1])
    cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
    box_outputs_topk = box_outputs

  return cls_outputs_topk, box_outputs_topk, classes, indices


def pre_nms(params,  cls_outputs, box_outputs):
    top_k = params.get("top_k",False)
    if top_k:
        # select topK purely based on scores before NMS, in order to speed up nms.
        cls_outputs, box_outputs, classes, indices = topk_class_boxes(
            params, cls_outputs, box_outputs)
    else:
        classes=None
    return box_outputs, cls_outputs,cls_outputs, classes
def postprocess_global(params, cls_outputs, box_outputs):
    '''
    faster nms global
    params: params_config
        +> num_classes = num_classes of cls_outputs = cls_outputs.shape[-1]
        +> top_k: select top_k before nms
        +> nms_configs:
            +> method: hard | gaussian
            +>max_nms_inputs: config max_mns_inputs select top_k 
            +>max_output_size: max_output_size select nms and pad if need.
            +> iou_thresh: iou detect two bboxes overlap
            +> score_thresh: score bbox keep
            +>sigma: if soft use sigma = sig*2 (sig :in paper)
            +>
    cls_outputs: bs,num_boxes,num_classes
    box_outputs: bs,num_boxes, 4
    '''
    boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)

    def single_batch_fn(element):
        return nms(params, element[0], element[1], element[2], True)
    nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
      single_batch_fn, [boxes, scores, classes])

    return nms_boxes, nms_scores, nms_classes, nms_valid_len