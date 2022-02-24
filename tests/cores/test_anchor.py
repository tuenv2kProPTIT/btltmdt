import os,sys,json
from pathlib import Path
import tensorflow as tf

os.system['CUDA_VISIBLE_DEVICES']=""
from tfdet.core.anchor.anchor_generator import AnchorGenerator, AnchorConfig
config_anchor_generator = dict(
    type='AnchorGenerator',
    strides=[10,],
    ratios=[1.,],
    scales=[1.,],
    base_sizes=[10],
)
config_anchor_generator = AnchorConfig(**config_anchor_generator)

anchor_generator = AnchorGenerator(config_anchor_generator)
all_anchors = anchor_generator.grid_anchors([(2, 2)])
expected_anchors = tf.convert_to_tensor([[-5., -5., 5., 5.], [5., -5., 15., 5.],
                                     [-5., 5., 5., 15.], [5., 5., 15., 15.]])

assert tf.math.reduce_mean(tf.math.abs(all_anchors[0] - expected_anchors)) < 1e-5


config_anchor_generator = dict(
    type='AnchorGenerator',
    strides=[(10,20)],
    ratios=[1.,],
    scales=[1.,],
    base_sizes=[10],
)
config_anchor_generator = AnchorConfig(**config_anchor_generator)

anchor_generator = AnchorGenerator(config_anchor_generator)
all_anchors = anchor_generator.grid_anchors([(2, 2)])
expected_anchors = tf.convert_to_tensor([[-5., -5., 5., 5.], [5., -5., 15., 5.],
                                     [-5., 15., 5., 25.], [5., 15., 15., 25.]])

assert tf.math.reduce_mean(tf.math.abs(all_anchors[0] - expected_anchors)) < 1e-5

print("pass all")