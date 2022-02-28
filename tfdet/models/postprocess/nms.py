import functools
from typing import List, Tuple

from absl import logging
import tensorflow as tf 
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
      tf.gather(classes, nms_top_idx) + CLASS_OFFSET, boxes.dtype)
  return nms_boxes, nms_scores, nms_classes, nms_valid_lens