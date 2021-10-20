# Adds information on the learning rate to the log.

import tensorflow as tf
from tensorflow import keras
import numpy as np

# NB: To modify log, use "on_epoch_end", not "on_epoch_begin".
#     The former will not work for CSVLogger, it gets overwritten.
class LRLog(keras.callbacks.Callback):
    
    def on_epoch_end(self,epoch,logs=None):
        
        # Get the non-decayed learning rate. This is the learning rate attribute
        # of the optimizer. It is affected by any LearningRateScheduler that we apply,
        # but it is not affected by the optional "decay" argument of the optimizer,
        # which applies an additional decay per batch.
        lr = self.model.optimizer.lr.numpy()
        logs['lr'] = lr
        
        # Get the number of iterations. Any decay applied to the learning rate is applied via
        # lr_t = lr / (1 + decay * iterations). This is potentially useful information if we need to restart
        # a run, and continue applying the same decay (so we will need to log how many iterations
        # we have reached).
        #iterations = self.model.optimizer.iterations.numpy()
        #logs['iterations'] = iterations
        
        # Get the "true" learning rate. There might be a decay applied,
        # so this learning rate is not necessarily equal to self.model.optimizer.lr.
        # This is the learning rate at the *end* of the epoch.
        #lr_t = self.model.optimizer._decayed_lr(tf.float32).numpy()
        #logs['lr_decayed'] = lr_t
        return