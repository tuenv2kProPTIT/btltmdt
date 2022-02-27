# Tensorflow Object Detection

## Design

1. Choose design like library not framework. But keep it simple for dummy, we are keep all API can load from register like timm, mmdet,...

### One-stage detector 

Onestage detector has three components:

1. backbones
2. neck
3. head

**Backbones: Design similarity with timm models, in the future we can load timm model pytorch to my models backbone.**

Backbones:

    +> Should implement: dummy_inputs to feed the checkpoints when loading.

    +> Call function backbones: has overide return_features for work with neck

Current:

1. [x] Support resnetv2 architecture
2. [] Convert weights from timm or torchvision 

**Neck: Design neck with input is feature pyramid of backbones**

Necks:

    +> Call function: receive inputs feature from backbones and proceduce pyramid features.

Current:

1. [x] Support FPN networks

**Head: Design flexible for many architectures with loss_cls and loss_bbox is fixed**

Heads:

    +> Should implement loss_fn: (cls_score, bbox_pred, anchor_level, target_boxes, target_labels, mask_labels) in the features level and batch_size instance level
    +> Should implement call function with inputs is pyramid features from necks and proceduce list[score_cls_level], list[score_bbox_level]

Current:

1. [x] Support all architecture use iou_loss and sigmoid_cls loss

---

## Simple Training

Think like ml project: To training some model, We should define datasets.

**Step 01: Dataset** : We keep dataset simple possible as possible, you should iterative over your datasets and convert it to our's format tfrecord. In the future we'll design some popular type of input dataset for cmd.

1. You have a dataset with: 
    - image_file: url absoulute to load image.
    - bbox_annotation: list[y_min, x_min, y_max, x_max]: keep in mind the order y is first.
    - label_annotation: list[{'id':id_categorical,'name':'name_categorical'}] : 0 \le **id** \l max_classes, name_categorical: str   (0<=id<max_caregorical)
2. You do whatever to group everything to list[{'image_path':image_file, 'bboxes':bbox_annotation, 'labels':label_annotation }].
3. from tfdet.dataio.tfrecord_utilts import convert_dataset_to_tfrecord
4. Call function convert_dataset_to_tfrecord(datasets_step2, num_shards=16,output_dir='directory')
5. Do again with valid datasets and testsets.

Note: In the future model can predict from embedded image numpy file for testsets. Now fake all annotation to [] for testsets.

**Step 02: Define pipeline load datasets**

```python3
from tfdet.dataio.pipeline import pipeline

steps_training_ds = [
    dict(
        name='InputReadRecordFiles',
        pattern_file=f'{directory}/*.tfrecord'
    ),
    dict(
        name='RandomResized',
        height=512,
        width=512,
        scale=(0.8,1.0),
        ratio= (0.75, 1.3333333333333333)
    ),
    dict(
        name='Normalize',
        mean= (0.485, 0.456, 0.406),
        std  = (0.229, 0.224, 0.225)
        p=1.
    )
]
train_ds = pipeline(
    steps_training_ds
)
def keep_only(example):
  return {
      'image':example['image'],
      'bboxes': example['bboxes'] * 512.,
      'labels':example['labels'],
      'mask':example['mask']
  }
train_ds = train_ds.map(lambda value:keep_only(value)).padded_batch(
    4,drop_remainder=True
)
# We reduce memory before batching by select some information usefull. And keep in mind bboxes don't have format [0-1] but format [0,w-1,0,h-1]
```

**Step 03: Define your model**

```python3
from tfdet.models.architectures.one_stage import OneStageModel,ConfigOneStage
from tfdet.models.callbacks
config_model = dict(
    backbone = dict(
        name='resnetv2',
        input_size=(512,512),
        nb_blocks=(3, 4, 6, 3),
        width_factor=1,
        pool_size=14,
        crop_pct=1.0,
    ),

    neck=dict(
        name='fpn',
        start_level=1,
        num_nb_ins=4,
        num_nb_outs=5,
        filters=256,
        add_extra_convs=True,
        extra_convs_on='on_input',
        relu_before_extra_convs=False, # in backbone we add norm-relu to top input and extra_convs_on='input' so we don't re-relu again.
    ),

    head = dict(
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
    )




)
model= OneStageModel(ConfigOneStage(**config))
model.compile(optimizer=tf.optimizers.Adam(1e-4))
callback=
model.fit(train_ds.prefetch(4), epochs=4,....)

```