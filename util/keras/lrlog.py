# Adds the current learning rate to the log, at the start of each epoch.

import tensorflow as tf
from tensorflow import keras

class LRLog(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs=None):
        logs.setdefault('lr',[]).append(self.model.optimizer.lr)
        return