from dataclasses import dataclass,field
import imp
import numpy as np
from tfdet.models.densen_heads.registry import register
from tfdet.models.config import HeadConfig
from tfdet.core.anchor.anchor_generator import AnchorConfig, AnchorGenerator
from tfdet.core.assigner.max_iou_assigner import MaxIOUAssignerConfig, MaxIOUAssigner
from tfdet.core.sampler.pseudo_sampler import PseudoSampler, PseudoSamplerConfig
from tfdet.core.bbox_encode.delta_wxyh import DeltaXYWH, DeltaXYWHConfig
from tfdet.utils.serializable import keras_serializable
from typing import Dict,Union
import tensorflow as tf 
from tfdet.models.layers.act_norm_conv import Conv2DNorm
from tfdet.utils.shape_utils import shape_list
from tfdet.utils.trick_tensor import gather_based_on_match
from tfdet.models.losses.build_loss import build_loss
@dataclass
class AnchorHeadConfig(HeadConfig):
    name='AnchorHead'
    
    anchor_config: Dict =  field(default_factory=dict) 
    assigner: Dict  =  field(default_factory=dict) 
    sampler : Dict =  field(default_factory=dict)
    bbox_encode: Dict =  field(default_factory=dict)
    num_heads: int = 4
    filters: int = 256
    act_cfg: Union[Dict, str]  = field(default_factory=lambda:'relu')
    norm_cfg: Union[Dict, str] = field(default_factory=lambda:'')
    num_classes: int = 80

    loss_cls : Dict = field(default_factory=lambda: {"name":'focalloss','use_sigmoid':True, 'loss_weight':1.0})
    loss_bbox: Dict = field(default_factory=lambda:{'name':'SmoothL1Loss','beta':1.0/9.0,'loss_weight':1.0})

    test_cfg : Dict = field(default_factory=lambda:{''})
    train_cfg :  Dict =field(default_factory=lambda: {
       
        'batch_size':16,

    })
    last_modified:str='25/02/2022'
@register
@keras_serializable
class AnchorHead(tf.keras.Model):
    cfg_class=AnchorHeadConfig
    def __init__(self, cfg:AnchorHeadConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg 
        self.anchor_generator = AnchorGenerator(AnchorConfig(**self.cfg.anchor_config))
        self.num_anchors=self.anchor_generator.num_base_anchors[0]
        if self.cfg.assigner.pop("name","max_iou_assigner") == 'max_iou_assigner':
            self.assigner = MaxIOUAssigner(MaxIOUAssignerConfig(**self.cfg.assigner))
        if self.cfg.sampler.pop("name","pseudo")=='pseudo':
            self.sampler = PseudoSampler(PseudoSamplerConfig(**self.cfg.sampler))
        if self.cfg.bbox_encode.pop("name","deltaxywh")=='deltaxywh':
            self.bbox_encode = DeltaXYWH(DeltaXYWHConfig(**self.cfg.bbox_encode))

        self.cal_loss_classes = build_loss(self.cfg.loss_cls)
        self.cal_loss_bboxes = build_loss(self.cfg.loss_bbox)
         
        self.head_cls = []
        self.head_bbox=[]
        for i in range(self.cfg.num_heads):
            self.head_cls.append(
                Conv2DNorm(
                    self.cfg.filters,
                    3,
                    strides=1,
                    padding='SAME',
                    act_cfg=self.cfg.act_cfg,
                    norm_cfg=self.cfg.norm_cfg,
                    order=('conv','norm','act')
                )
            )
            self.head_bbox.append(
                Conv2DNorm(
                    self.cfg.filters,
                    3,
                    strides=1,
                    padding='SAME',
                    act_cfg=self.cfg.act_cfg,
                    norm_cfg=self.cfg.norm_cfg,
                    order=('conv','norm','act')
                )
            )
        self.head_cls.append(
            tf.keras.layers.Conv2D(
                self.num_anchors * self.cfg.num_classes,3,
                padding='SAME',kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
                bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),)
        )
        self.head_bbox.append(
            tf.keras.layers.Conv2D(
                self.num_anchors *4, 3 ,
                padding='SAME',kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
                bias_initializer=tf.random_normal_initializer(stddev=0.01),)
        )



    def loss_fn(self, cls_score, bbox_pred, target_boxes, target_labels, mask_labels):
        shape_list_feature = [shape_list(i) for i in cls_score]
        anchors = self.anchor_generator.grid_priors([ shape[-3:-1] for shape in shape_list_feature]) 
        loss_dict={'cls_loss':[],'bbox_loss':[]}
        for level in range(len(cls_score)):
            # total_loss_box=[]
            # total_loss_cls=[]
            loss_bbox, loss_cls = tf.vectorized_map(self.loss_fn_reduce_on_features, cls_score[level],bbox_pred[level],anchors[level], target_boxes,target_labels, mask_labels )
            # for batch in range(self.cfg.train_cfg['batch_size']):
            #     loss_bbox, loss_cls = self.loss_fn_reduce_on_features(
            #         cls_score[level][batch,...], bbox_pred[level][batch,...],anchors[level], target_boxes[batch,...],target_labels[batch,...],mask_labels[batch,...]
            #     )
            #     total_loss_box.append(loss_bbox)
            #     total_loss_cls.append(loss_cls)
            loss_dict['cls_loss'].append(tf.math.reduce_sum(loss_bbox) )
            loss_dict['bbox_loss'].append(tf.math.reduce_sum(loss_cls))
        loss_dict[f'cls_loss'] = sum(loss_dict['cls_loss']) / float(self.cfg.train_cfg['batch_size'])
        loss_dict[f'bbox_loss']= sum(loss_dict['bbox_loss']) / float(self.cfg.train_cfg['batch_size'])
        return loss_dict

    def loss_fn_reduce_on_features(self, cls_score, bbox_pred, anchor_level, target_boxes, target_labels, mask_labels):
        """
        cls_score: w,h,num_anchors * num_classes
        bbox_pred: w,h,num_anchors * 4
        anchor_level: 1,m,4
        
        function work on batch_size is 1: reduce_memory but incresea time to training.
        """
        target_labels=tf.reshape(target_labels,[-1,1])
        mask_labels = tf.reshape(mask_labels, [-1,])
        shape_list_feature=shape_list(cls_score)
        cls_score = tf.reshape(cls_score,[1* shape_list_feature[0] * shape_list_feature[1] * self.num_anchors,self.cfg.num_classes])
        # bs,M,num_classes
        bbox_pred = tf.reshape(bbox_pred, [1* shape_list_feature[1] * shape_list_feature[0] * self.num_anchors, 4 ])
        # bs,M,4
        # print("anchor level", anchor_level)
        # print("mask_labels",mask_labels)
        index_matching  = self.assigner.match(anchors=anchor_level, targets=target_boxes, ignore_tagets=mask_labels)

        index_matching = self.sampler.sampler(index_matching)
        index_matching = tf.stop_gradient(index_matching)
        # print("index_matching",index_matching)
        # -2 for ignore,-1 negative, index for positive
        # print("taget",target_boxes)
        matched_gt_boxes = gather_based_on_match(
            target_boxes,
            tf.zeros(4),
            tf.zeros(4),
            index_matching,
            name_ops=self.cfg.train_cfg.get('gather_type','gather_normal')
        )
        matched_gt_boxes=tf.stop_gradient(matched_gt_boxes)
        # print("matched",matched_gt_boxes )
        matched_reg_targets =self.bbox_encode.encode(
            matched_gt_boxes,
            anchor_level
        )
        mask_reg_targets = tf.where(index_matching >=0, 1, 0) 
        total_matched = tf.maximum(0.000001,tf.cast(tf.reduce_sum(mask_reg_targets),tf.float32))
        matched_gt_classes = gather_based_on_match(
            target_labels,
            tf.constant([-1],tf.int32),
            tf.constant([-1],tf.int32),
            index_matching
        ) 
        matched_gt_classes=tf.stop_gradient(matched_gt_classes)
        mask_classes_tagets = tf.where(index_matching >= -1, 1, 0)
        loss_bbox = self.cal_loss_bboxes.compute_loss(
            bbox_pred,
            matched_reg_targets,
            mask_reg_targets
        ) / (total_matched  )
        loss_cls = self.cal_loss_classes.compute_loss(
            cls_score,
            matched_gt_classes,
            mask_classes_tagets
        ) / (total_matched)

        return loss_bbox, loss_cls





        
        

    def call(self, inputs, training=None):
        outs=[]
        for input in inputs:
            outs.append(
                self.forward_single(
                    input, training=training
                )
            )
        outs=tuple(map(list, zip(*outs)))
        return outs 
    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, x,training=None):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = x
        bbox_pred=x 
        for layer in self.head_cls:
            cls_score = layer(cls_score,training=training)
        for layer in self.head_bbox:
            bbox_pred=layer(bbox_pred,training=training)
        
        return cls_score, bbox_pred
    @tf.autograph.experimental.do_not_convert
    def simple_infer(self, cls_scores, bbox_preds):
        '''
        cls_scores: [(Bs,h_level,w_level,num_classes * num_anchors),... ]
        bbox_preds: [(Bs,h_level,w_level,4 * num_anchors,), ...]
        '''

        target_labels=tf.reshape(target_labels,[-1,1])
        mask_labels = tf.reshape(mask_labels, [-1,])
        shape_list_feature=shape_list(cls_score)
        cls_score = tf.reshape(cls_score,[1* shape_list_feature[0] * shape_list_feature[1] * self.num_anchors,self.cfg.num_classes])
        # bs,M,num_classes
        bbox_pred = tf.reshape(bbox_pred, [1* shape_list_feature[1] * shape_list_feature[0] * self.num_anchors, 4 ])


