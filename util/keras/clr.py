# Based on code from keras_contrib, at https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_lr.py

import tensorflow as tf
from tensorflow import keras

# Reference: 
class CyclicLearningRate(keras.callbacks.Callback):
    def __init__(self, base_lr=1.0e-3, max_lr=5.0e-3, step_size=2.0e3, mode='triangular',gamma=1., scale_fn=None, scale_mode='cycle'):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = 'triangular'
        self.gamma = 1.
        self.history = {}
        
        if scale_fn is None:
            if(self.mode == 'triangular'):
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif(self.mode == 'traingular2'):
                self.scale_fn = lambda x: 1. / (2.**(x-1))
                self.scale_mode = 'cycle'
            elif(self.mode == 'exp_range'):
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'     
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
            
        self.clr_iterations = 0
        self.trn_iterations = 0
        
    def _reset(self):
        self.clr_iterations = 0
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2. * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if(self.scale_mode == 'cycle'):
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x) * self.scale_fn(cycle))
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x) * self.scale_fn(self.clr_iterations))
        
    def on_train_begin(self,logs={}):
        logs = logs or {}
        if(self.clr_iterations == 0):
            keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            keras.backend.set_value(self.model.optimizer.lr, self.clr())
            
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1
        self.trn_iterations += 1
        keras.backend.set_value(self.model.optimizer.lr, self.clr())
        self.history.setdefault(
            'lr',[]).append(
                self.model.optimizer.lr
        )
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        for k,v in logs.items():
            self.history.setdefault(k, []).append(v)
            
    def on_epoch_end(self,epoch, logs=None):
        logs = logs or {}
        logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)
        