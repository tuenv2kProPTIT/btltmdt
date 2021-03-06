import imp
from tfdet.core.config import CoreConfig
from dataclasses import  dataclass,field
from typing import List, Tuple
import tensorflow as tf
from tfdet.utils.constants import EPSILON
@dataclass
class DeltaXYWHConfig(CoreConfig):
    last_modified = '25/02/2022'
    name = 'bbox_encoder'
    
    scale_factors:Tuple[int] =field(default_factory=lambda: [1., 1., 1., 1.])

def get_center_coordinates_and_sizes(box_corners, scope='None'):
    """Computes the center coordinates, height and width of the boxes.
    Args:
      scope: name scope of the function.
    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    # with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]
class DeltaXYWH():
    cfg_class=DeltaXYWHConfig
    def __init__(self, cfg:DeltaXYWHConfig, *args, **kwargs) -> None:
        self.cfg = cfg 
    
    def encode(self, bboxes, anchors):
        y,x,h,w =get_center_coordinates_and_sizes(bboxes)
        y_a,x_a,h_a,w_a = get_center_coordinates_and_sizes(anchors)

        h_a += EPSILON
        w_a += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (x - x_a) / w_a
        ty = (y - y_a) / h_a
        tw = tf.math.log(w / w_a)
        th = tf.math.log(h / h_a)
        
        ty *= self.cfg.scale_factors[0]
        tx *= self.cfg.scale_factors[1]
        th *= self.cfg.scale_factors[2]
        tw *= self.cfg.scale_factors[3]
        return tf.transpose(tf.stack([ty, tx, th, tw]))
    def decode_batch(self, rel_codes, anchors):
        # anchors: m,4
        # rel_codes: bs,m,4
        ycenter_a, xcenter_a, ha, wa =  get_center_coordinates_and_sizes(anchors)
        ycenter_a = tf.expand_dims(ycenter_a,0)
        xcenter_a = tf.expand_dims(xcenter_a, 0)
        ha = tf.expand_dims(ha,0)
        wa= tf.expand_dims(wa,0)
        ty, tx, th, tw = tf.unstack(rel_codes, axis=-1)
        ty /= self.cfg.scale_factors[0]
        tx /= self.cfg.scale_factors[1]
        th /= self.cfg.scale_factors[2]
        tw /= self.cfg.scale_factors[3] # bs,m
        w = tf.exp(tw) * wa 
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2. #bs,m

        return tf.stack([ymin, xmin, ymax, xmax],-1) # bs,m,4


    def decode(self, rel_codes, anchors):
        ycenter_a, xcenter_a, ha, wa =  get_center_coordinates_and_sizes(anchors)
        ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
        ty /= self.cfg.scale_factors[0]
        tx /= self.cfg.scale_factors[1]
        th /= self.cfg.scale_factors[2]
        tw /= self.cfg.scale_factors[3]
        w = tf.exp(tw) * wa 
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
