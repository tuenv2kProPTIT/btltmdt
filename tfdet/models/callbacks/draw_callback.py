import enum
import tensorflow as tf 
from tfdet.core.visualize.image import make_target,plot_images
import os 
class DrawCallBack(tf.keras.callbacks.Callback):
    def __init__(self, test_ds,total_sample=1,log_dir="./logs_draw", **kwargs):
        super().__init__(**kwargs)
        os.makedirs(log_dir, exist_ok=True)
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
            target = make_target(sample['bboxes'],sample['labels'],sample['mask'])
            plot_images(image, target, fname=os.path.join(log_epoch,f"{index}_label.jpg"))
    def on_epoch_end(self,  epoch, logs=None):
        epoch = epoch + 1
        log_epoch = os.path.join(self.log_dir,f"{epoch}")
        for index,sample in enumerate(self.batch_sample_test):
            out = self.model.test_step(sample)
            target = self.model.simple_infer(*out)
            image=sample['image'].numpy()
            plot_images(image, target, fname=os.path.join(log_epoch,f"{index}_predict.jpg"))

        

        
        