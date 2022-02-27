from dataclasses import dataclass, field

from typing import Dict
import tensorflow as tf 
from tfdet.utils.serializable import keras_serializable
from tfdet.models.backbones.registry import get_backbone
from tfdet.models.densen_heads.registry import get_densen_head
from tfdet.models.necks.registry import get_neck
@dataclass
class ConfigOneStage:
    name='OneStage'
    url = 'None'
    last_modified='25/02/2022'

    backbone: Dict = field(default_factory=lambda : dict(
        name='resnetv2',
        input_size=(512,512),
        nb_blocks=(3, 4, 6, 3),
        width_factor=3,
        pool_size=14,
        crop_pct=1.0,
    ))


    neck : Dict = field(default_factory= lambda : dict(
        name='fpn',
        num_nb_ins=4,
        num_nb_outs=5,
        filters=256,
        add_extra_convs=True,
        extra_convs_on='on_input',
        relu_before_extra_convs=False,
    ))

    head : Dict = field(default_factory= lambda : dict(
        name="AnchorHead",
        anchor_config=dict(
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0,),
        assigner=dict(
            name='max_iou_assigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
        ),
        sampler=dict(
            name='pseudo'
        ),
        bbox_encode=dict(
            name='deltaxywh',
            scale_factors=[1.,1.,1.,1.]),
        act_cfg='relu',
        num_classes=20,

        loss_cls= {"name":'focalloss','use_sigmoid':True, 'loss_weight':1.0},
        loss_bbox={'name':'SmoothL1Loss','beta':1.0/9.0,'loss_weight':1.0},

        train_cfg=dict(
            batch_size=4
        )
    ))

    class_cfg : Dict = field(default_factory= lambda : {
        0:"aeroplane", 1:"bicycle",2:"bird",3:"boat",
        4:"bottle", 5:"bus", 6:"car", 7:"cat",8:"chair",
        9:"cow",10: "diningtable",11: "dog",12: "horse",
        13:"motorbike", 14:"person",15: "pottedplant",
        16:"sheep",16: "sofa",18: "train",19: "tvmonitor"
    })



class OneStageModel(tf.keras.Model):
    cfg_class=ConfigOneStage
    def __init__(self, cfg:ConfigOneStage, *args, **kwargs) -> None:
        super().__init__()
        self.cfg= cfg 
        self.backbone = get_backbone(self.cfg.backbone)
        self.neck = get_neck(self.cfg.neck)
        self.head = get_densen_head(self.cfg.head)
    def call(self, inputs, training=None):
        features  =self.backbone(inputs, training=training,return_features=True)
        neck=self.neck(features, training=training)
        cls_score,bbox_score = self.head(neck, training=training)
        return cls_score, bbox_score
    def get_config(self):
        cfg=dict(
            backbone=self.backbone.get_config(),
            neck=self.neck.get_config(),
            head=self.head.get_config()
        )
        return cfg
    @property
    def dummy_inputs(self) -> tf.Tensor:
        return self.backbone.dummy_inputs

    def train_step(self, data):
        image  =data['image']
        bboxes = data['bboxes']
        labels = data['labels']
        mask_label = data['mask']
        with tf.GradientTape() as tape:
            cls_score, bbox_score = self(image, training=True)
            loss_dict=self.head.loss_fn( cls_score, bbox_score, bboxes, labels, mask_label)
            loss_dict['loss_additional'] = sum(self.losses)
            loss = sum(loss_dict.values())
            trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss_dict