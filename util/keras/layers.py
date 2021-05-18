# Here we can code up custom layers to implement in models.py

# See this reference
# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb (in use)
# This code also appears to match tf.keras source code for ResNet50 implementation.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform

class IdentityBlock(layers.Layer):
    """
    Implementation of the ResNet identity block.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    def __init__(self, f, filters, stage, block, normalization=True, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs) # god knows what this does...
        
        # retrieve attributes
        self.f = f
        self.filters = filters
        self.F1, self.F2, self.F3 = filters
        self.stage = stage
        self.block = block
        self.normalization = normalization
        
        conv_name_base =  'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'
        
        self.F1, self.F2, self.F3 = self.filters

        # create the inner layers here
        self.conv1 = Conv2D(filters = self.F1, kernel_size = (1, 1),\
                            strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))
        self.conv2 = Conv2D(filters = self.F2, kernel_size = (self.f, self.f),\
                            strides = (1,1), padding = 'same',  name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))
        self.conv3 = Conv2D(filters = self.F3, kernel_size = (1, 1),\
                            strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))
        
        self.bn1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')
        self.bn2 = BatchNormalization(axis = 3, name = bn_name_base + '2b')
        self.bn3 = BatchNormalization(axis = 3, name = bn_name_base + '2c')
        
        self.add = Add()

    def call(self, inputs):
        X_shortcut = inputs
        X = self.conv1(inputs)
        if(self.normalization): X = self.bn1(X)
        X = tf.nn.relu(X)
        
        X = self.conv2(X)
        if(self.normalization): X = self.bn2(X)
        X = tf.nn.relu(X)

        X = self.conv3(X)
        if(self.normalization): X = self.bn3(X)
        X = tf.nn.relu(X)

        X = self.add([X, X_shortcut])
        X = tf.nn.relu(X)
        return X
    
    def get_config(self):
        config = super(IdentityBlock, self).get_config()
        config.update(
            {
                'f':self.f,
                'filters': self.filters,
                'stage': self.stage,
                'block': self.block,
                'normalization': self.normalization,
            }
        )
        return config

class ConvolutionBlock(layers.Layer):
    """
    Implementation of the ResNet convolutional block.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    def __init__(self, f, filters, stage, block, s=2, normalization=True, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs) # god knows what this does...
        
        # retrieve attributes
        self.f = f
        self.filters = filters
        self.F1, self.F2, self.F3 = filters
        self.stage = stage
        self.block = block
        self.s = s
        self.normalization = normalization
        
        conv_name_base =  'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'
        
        self.F1, self.F2, self.F3 = self.filters

        # create the inner layers here
        self.conv1 = Conv2D(filters = self.F1, kernel_size = (1, 1),\
                            strides = (self.s,self.s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))
        self.conv2 = Conv2D(filters = self.F2, kernel_size = (self.f, self.f),\
                            strides = (1,1),           padding = 'same',  name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))
        self.conv3 = Conv2D(filters = self.F3, kernel_size = (1, 1),\
                            strides = (1,1),           padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))
        self.conv1b = Conv2D(filters = self.F3, kernel_size = (1, 1),\
                             strides = (self.s,self.s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))
        
        self.bn1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')
        self.bn2 = BatchNormalization(axis = 3, name = bn_name_base + '2b')
        self.bn3 = BatchNormalization(axis = 3, name = bn_name_base + '2c')
        self.bn1b = BatchNormalization(axis = 3, name = bn_name_base + '1')

        self.add = Add()

    def call(self, inputs):
        X_shortcut = inputs
        X = self.conv1(inputs)
        if(self.normalization): X = self.bn1(X)
        X = tf.nn.relu(X)
        
        X = self.conv2(X)
        if(self.normalization): X = self.bn2(X)
        X = tf.nn.relu(X)

        X = self.conv3(X)
        if(self.normalization): X = self.bn3(X)
        X = tf.nn.relu(X)
        
        X_shortcut = self.conv1b(X_shortcut)
        if(self.normalization): X_shortcut = self.bn1b(X_shortcut)
        X_shortcut = tf.nn.relu(X_shortcut)
        
        X = self.add([X, X_shortcut])
        X = tf.nn.relu(X)
        return X
    
    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                'f':self.f,
                'filters': self.filters,
                'stage': self.stage,
                'block': self.block,
                's':self.s,
                'normalization': self.normalization,
            }
        )
        return config

class ImageScaleBlock(layers.Layer):
    def __init__(self, new_shape, normalization=True, name_prefix='scaled_input_', method='nearest', **kwargs):
        super(ImageScaleBlock, self).__init__(**kwargs) # god knows what this does...
        
        # retrieve attributes
        self.new_shape = new_shape
        self.normalization = normalization
        self.name_prefix = name_prefix
        self.method = method
        
    def call(self, inputs):
        channels = len(inputs)
        scaled_inputs = [tf.image.resize(x,self.new_shape,name=self.name_prefix+str(i), method=self.method) for i,x in enumerate(inputs)]            
        if(self.normalization):
            integrals_old = [tf.math.reduce_sum(x,axis=[1,2]) for x in inputs]
            integrals_new = [tf.math.reduce_sum(x,axis=[1,2]) for x in scaled_inputs]        

            # We need to get rid of any zeros in integrals_new, to avoid creating nan's.
            # See https://stackoverflow.com/a/57630185
            integrals_new = tf.where(tf.equal(integrals_new, 0), tf.ones_like(integrals_new), integrals_new)        
            
            # normalizations are ratio of integrals (for each image in the batch)
            normalizations = [tf.math.divide(integrals_old[i],integrals_new[i]) for i in range(channels)]
            # now fix the dimensions of normalizations, to properly broadcast. Dims should be (batch, eta, phi, channel),
            # so we must insert 2 axes in the middle as we currently have (batch, channel).
            # These new axes will be of size one, will be taken care of by broadcasting.
            for i in range(channels):
                for j in range(2): normalizations[i] = tf.expand_dims(normalizations[i],axis=1)
            scaled_inputs = [tf.math.multiply(normalizations[i],scaled_inputs[i]) for i in range(channels)]
        X = tf.concat(values=scaled_inputs, axis=3, name='concat')
        return X
    
    def get_config(self):
        config = super(ImageScaleBlock, self).get_config()
        config.update(
            {
                'new_shape':self.new_shape,
                'normalization': self.normalization,
                'name_prefix': self.name_prefix
            }
        )
        return config
    
# A simple layer for normalizing a tensor's integral.
class NormalizationBlock(layers.Layer):
    def __init__(self, axes, scaling=1.0, name_prefix='normalization_', **kwargs):
        super(NormalizationBlock, self).__init__(**kwargs) # god knows what this does...
        
        # retrieve attributes
        self.axes = axes
        self.scaling = scaling
        self.name_prefix = name_prefix
        
    def call(self, inputs):
        
        integral = tf.math.reduce_sum(inputs,axis=self.axes)
        integral = tf.where(tf.equal(integral, 0), tf.ones_like(integral), integral) # avoid division by zero      
        for i in range(len(self.axes)):
            integral = tf.expand_dims(integral,axis=1)
        return tf.math.multiply(self.scaling, tf.math.divide(inputs,integral))
            
    def get_config(self):
        config = super(NormalizationBlock, self).get_config()
        config.update(
            {
                'axes':self.axes,
                'scaling':self.scaling,
                'name_prefix': self.name_prefix
            }
        )
        return config