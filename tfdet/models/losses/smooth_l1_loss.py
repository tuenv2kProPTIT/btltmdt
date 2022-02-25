import tensorflow as tf 
from dataclasses import dataclass



@dataclass
class SmoothL1LossConfig:
    beta:float=1.0
    loss_weights: float=1.0
    name='SmoothL1Loss'
    last_modified='25/02/2022'


class SmoothL1Loss:
    cfg_class = SmoothL1LossConfig
    def __init__(self, cfg: SmoothL1LossConfig, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg 
    
    def compute_loss(self, pred, target, weights):
        diff = tf.math.abs(pred - target)
        loss = tf.where(diff < self.cfg.beta, 0.5 * diff * diff / self.cfg.beta,
                       diff - 0.5 * self.cfg.beta)
        
        loss = tf.math.reduce_sum(loss,-1)
        weights = tf.cast(weights,tf.float32) 
        weights = tf.reshape(weights,[-1,])
        loss= loss * weights
        return tf.math.reduce_sum(loss) * self.cfg.loss_weights

        



