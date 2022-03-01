import tensorflow as tf 
from dataclasses import dataclass
from tfdet.utils.shape_utils import shape_list
from tfdet.models.losses.build_loss import register

@dataclass
class FocalLossConfig:
    gamma:float=1.5
    alpha: float=0.25
    use_sigmoid:bool  = True 
    loss_weight:float = 0.1
    name:str='FocalLoss'
    last_modified:str='25/02/2022'

@tf.function(experimental_relax_shapes=True)
def focal_loss_funtion(pred, target, alpha = 0.25, gamma = 2., label_smoothing = 0.):
    pred_prob = tf.nn.sigmoid(pred)
    p_t = ((1-target )* pred_prob) + (target * (1 - pred_prob))
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor =  tf.pow(p_t,gamma)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
    loss_without_weights= alpha_factor * modulating_factor * ce
    return tf.math.reduce_sum(loss_without_weights,axis=-1)
@register
class FocalLoss:
    cfg_class = FocalLossConfig
    def __init__(self, cfg: FocalLossConfig, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg 
    
    def compute_loss(self, pred, target, weights):
        num_classes = shape_list(pred)[-1]
        target = tf.one_hot(tf.reshape(target,[-1,]),  depth=num_classes)
        loss_cls =focal_loss_funtion(pred, target) 
        weights = tf.reshape(weights,(-1,))
        loss_cls = loss_cls * tf.cast(weights,tf.float32) * self.cfg.loss_weight
        return tf.math.reduce_sum(loss_cls)

        



