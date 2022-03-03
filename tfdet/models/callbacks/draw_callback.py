import enum
import tensorflow as tf 
from tfdet.core.visualize.image import make_target,plot_images
import os 
from tfdet.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
class DrawCallBack(tf.keras.callbacks.Callback):
    def __init__(self, test_ds,total_sample=1,log_dir="./logs_draw",image_net_transform=True, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(log_dir, exist_ok=True)
        self.image_net_transform=image_net_transform
        self.batch_sample_test = list(test_ds.take(total_sample))
        self.log_dir=log_dir
    def on_epoch_begin(self,epoch, logs=None):
        epoch =epoch + 1 
        log_epoch = os.path.join(self.log_dir,f"{epoch}")
        os.makedirs(os.path.join(self.log_dir,f"{epoch}"), exist_ok=True)
        if epoch !=1:
            return
        for index,sample in enumerate(self.batch_sample_test):
            image=sample['image'].numpy()
            if self.image_net_transform:
                image = image * 255. * np.array(IMAGENET_DEFAULT_STD).reshape([1,1,1,3]) + np.array(IMAGENET_DEFAULT_MEAN).reshape([1,1,1,3]) * 255.
            target = make_target(sample['bboxes'],sample['labels'],sample['mask'])
            plot_images(image, target, fname=os.path.join(log_epoch,f"{index}_label.jpg"))
    def on_epoch_end(self,  epoch, logs=None):
        epoch = epoch + 1
        log_epoch = os.path.join(self.log_dir,f"{epoch}")
        for index,sample in enumerate(self.batch_sample_test):
            out = self.model.test_step(sample)
            target = self.model.simple_infer(*out)
            image=sample['image'].numpy()
            if self.image_net_transform:
                image = image * 255. * np.array(IMAGENET_DEFAULT_STD).reshape([1,1,1,3]) + np.array(IMAGENET_DEFAULT_MEAN).reshape([1,1,1,3]) * 255.
            plot_images(image, target, fname=os.path.join(log_epoch,f"{index}_predict.jpg"))

        

        
        