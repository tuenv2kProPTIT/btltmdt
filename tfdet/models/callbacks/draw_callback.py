import tensorflow as tf 


class DrawCallBack(tf.keras.callbacks.Callback):
    def __init__(self, test_ds,total_sample=1, **kwargs):
        super().__init__(**kwargs)

        self.batch_sample_test = list(test_ds.take(total_sample))

    
    def on_epoch_end(self,  epoch, logs=None):
        epoch = epoch + 1
        