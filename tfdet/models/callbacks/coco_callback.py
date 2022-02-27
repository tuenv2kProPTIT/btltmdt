from __future__ import annotations
import enum
import tensorflow as tf 
from dataclasses import dataclass,asdict,field
from typing import Dict,Tuple
from tfdet.dataio.pipiline import pipeline
import time
from tqdm import tqdm
import numpy as np
import os,json
@dataclass
class CocoCallBackConfig:
    name : str= 'cococallbacks'
    last_modified: str ='27/02/2022'
    use_model: bool = False
    pipeline_evalset: Tuple[Dict] = field(default_factory= lambda : [])
    batch_size : int = 1
    map_slice : Tuple[float,] = field(default_factory= lambda : (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95))
    log_dir : str = "./logs"
class CocoCallBack(tf.keras.callbacks.Callback):
    cfg_class=CocoCallBackConfig
    def __init__(self, cfg : CocoCallBackConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.cfg = cfg   
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.ds = pipeline(self.cfg.pipeline_evalset).padded_batch(self.cfg.batch_size)

    def on_epoch_end(self,  epoch, logs=None):
        epoch = epoch + 1
        all_detections=[]
        map_id = {}
        tf.print("Running evaluated cococalback\n")
        start = time.time()
        for step,ds in tqdm(enumerate(self.ds)):
            total_sample = len(ds[ next(ds.keys() ) ] )
            ds_dict = self.model.test_step(ds)
            list_id = ds_dict.pop("id")
            for id in list_id:
                if id not in map_id:
                    map_id[id] = len(map_id)
                    all_detections.append({})
            for o in range(total_sample):
                id = list_id[o]
                vt = map_id[id] 
                for k in ds_dict.keys():
                    v=ds_dict[k][o]
                    if hasattr(v,'numpy'):
                        v=v.numpy()
                    if k not in all_detections[vt]:
                        all_detections[vt][k] = [v]
                    else:
                        all_detections[vt][k].append(v)
        all_detections = [{k:np.concatenate(v) for k,v in all_detections[i].items()} for i in range(len(all_detections))]
        end = time.time()
        tf.print(f"Predict done affter {end-start} second")
        aps = [self.compute_map(all_detections,threshold_iou=i ) for i in self.cfg.map_slice]
        mans = sum([i[0] for i in aps]) / len(self.cfg.map_slice)
        logs = {
            f'map_slice_{str(self.cfg.map_slice)}':mans,
        }
        with open(os.path.join(self.cfg.log_dir,f'epoch_coco_{epoch}.json'),"w+") as f:
            json.dump(logs,f)
        print("compute done")
        print(logs)
    def compute_map(self,all_detections, threshold_iou ):
        all_metrics={}
        for class_cfg in self.class_cfg:
            class_name = self.class_cfg[class_cfg]
            false_positives = []
            true_positives = []
            scores = []
            num_annotations = 0.0
            for step,sample in enumerate(all_detections):
                bboxes = sample['bboxes']
                label = sample['labels']
                mask = sample['mask']
                bboxes = bboxes[mask]
                label = label[mask]
                pred_bbox = sample['pred_bboxes'] #num-detect,4
                pred_label = sample['pred_labels']# num-detect,
                score = sample['pred_scores'] # num-detect-
                idx = np.where(score >= threshold_iou, 1, 0)
                score = score[idx]
                pred_bbox = pred_bbox[idx]
                pred_label = label[idx]
                detections=[]
                annotations=[]
                for pred_b,pred_l,pred_sc in zip(pred_bbox, pred_label, score):
                    if pred_l == class_cfg:
                        detections.append(
                            pred_b + [pred_l,] + [pred_sc,]
                        )
                detections=sorted(detections,key=lambda x:-1*x[-1]) # sort by score
                for label_b, label_l in zip(bboxes, label):
                    if label_l == class_cfg:
                        annotations.append(
                            label_b + [label_l,]
                        )
                num_annotations=num_annotations+len(annotations)

                scr, fp, tp = self.check_if_true_or_false_positive(annotations,detections )
                scores+= scr 
                false_positives += fp 
                true_positives += tp
            if num_annotations == 0:
                all_metrics[class_name] = 0, 0,[],[]
                continue
            false_positives = np.array(false_positives)
            true_positives = np.array(true_positives)
            scores = np.array(scores)

            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            average_precision = _compute_ap(recall, precision)
            all_metrics[class_name] = average_precision,num_annotations, precision, recall
        
        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations, _, _) in all_metrics.items():
            if num_annotations > 0:
                present_classes += 1
                precision += average_precision
        mean_ap = precision / present_classes
        return mean_ap, all_metrics
    def compute_overlaps(self, annotations, detections):
        annotation = np.array(annotations)[:,:4]
        detection = np.array(detections)[:,:4]
        return box_iou(annotation, detection)
    def check_if_true_or_false_positive(self,annotation, detections, iou_threshold):
        '''https://www.kaggle.com/its7171/map-understanding-with-code-and-its-tips/notebook'''

        scores = []
        false_positives = []
        true_positives = []
        detected_annotations = []
        overlap = self.compute_overlaps( detections,annotation) #m_detection,n_annotation
        for idx,d in enumerate(detections):
            scores.append(d[5])
            if len(annotation) == 0:
                false_positives.append(1)
                true_positives.append(0)
                continue
            overlap_d = overlap[idx,:]
            assigned_annotation = np.argmax(overlap_d)
            max_overlap = overlap_d[assigned_annotation]
            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives.append(0)
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives.append(1)
                true_positives.append(0)
        return scores, false_positives, true_positives


    def set_model(self, model):
        self.model  = model 
        self.class_cfg = self.model.get_config()['class_cfg']
    def get_config(self):
        return asdict(self.cfg)

def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height

def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union

def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou
def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap