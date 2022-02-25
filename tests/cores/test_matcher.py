from tfdet.core.assigner.max_iou_assigner import *
import numpy as np
import tensorflow.compat.v1 as tf
import os,sys,json
from pathlib import Path

cfg = dict(pos_iou_thr=3.,neg_iou_thr=2.,force_match_for_each_row=False)
cfg=MaxIOUAssignerConfig(**cfg)
matcher = MaxIOUAssigner(cfg)
similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.float32)
similarity=tf.convert_to_tensor(similarity)
valid_rows=tf.ones(3)
match = matcher._match(similarity,valid_rows)
# match=<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 2, -1, -1,  0, -1], dtype=int32)>

cfg = dict(pos_iou_thr=3.,neg_iou_thr=2.,force_match_for_each_row=True)
cfg=MaxIOUAssignerConfig(**cfg)
matcher = MaxIOUAssigner(cfg)
similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.float32)
similarity=tf.convert_to_tensor(similarity)
valid_rows=tf.ones(3)
match = matcher._match(similarity,valid_rows)
# match=<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 2,  1, -1,  0, -1], dtype=int32)>
cfg = dict(pos_iou_thr=3.,neg_iou_thr=2.,force_match_for_each_row=True)
cfg=MaxIOUAssignerConfig(**cfg)
matcher = MaxIOUAssigner(cfg)
similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [0, 0, 0, 0, 0],
                           [3, 0, -1, 2, 0],
                           [0, 0, 0, 0, 0]], dtype=np.float32)
similarity=tf.convert_to_tensor(similarity)
valid_rows=tf.convert_to_tensor([1,1,0,1,1],dtype=tf.int32)
match = matcher._match(similarity,valid_rows)
# match=<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 3,  1, -1,  0, -1], dtype=int32)>