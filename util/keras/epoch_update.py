# Updates the "initial_epoch" attribute of a network at the end of each epoch.
# If checkpointing the network, this will make re-training to a fixed, total number of epochs easy,
# as the network will have a memory of how many epochs it was already trained for.
# We may alternatively achieve this functionality by checking a log file associated with the network,
# but saving the info into the model itself keeps things all in one file.

import tensorflow as tf
from tensorflow import keras


# TODO: Does not work, initial_epoch is known to KerasRegressor but is *not* an attribute of model (tf.keras.models.Sequential, or other)
class EpochUpdate(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        print('self.model.initial_epoch = {}'.format(self.model.initial_epoch))
        keras.backend.set_value(self.model.initial_epoch, epoch)
        return