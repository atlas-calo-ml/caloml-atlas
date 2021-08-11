# Adds the current learning rate to the log, at the start of each epoch.

import tensorflow as tf
from tensorflow import keras
import numpy as np

def LearningRateSchedule(mode='exp', gamma=0.1, offset=0):

    if(mode == 'exp'):
        def scheduler(epoch,lr):
            if(epoch <= offset): return lr
            else: return lr * tf.math.exp(-gamma)    
    else:
        def scheduler(epoch,lr):
            return lr
    return tf.keras.callbacks.LearningRateScheduler(scheduler)