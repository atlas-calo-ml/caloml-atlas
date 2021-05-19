# More experimental models / less organized code that we don't want cluttering
# up our models.py file.

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Add, Concatenate, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from string import ascii_lowercase

# Custom layers.
from util.keras.layers import *

# A network that uses a combo of CNN and "simpler" functionalities.
# For simplicity, it just takes CNN-style input (images).
class exp_cnn_model():
    def __init__(self, input_shape, kernel=(3,3), lr=5e-5, dropout=-1., augmentation=True, normalization=True):
        self.lr = lr
        self.input_shape = input_shape
        self.kernel = kernel
        self.dropout = dropout
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock,
            'NormalizationBlock':NormalizationBlock
        } 
        
    def model(self):
        input_shape = self.input_shape
        kernel = self.kernel
        lr = self.lr
        dropout = self.dropout
        augmentation = self.augmentation
        normalization = self.normalization
        
        # Input images from all calorimeter layers.
        input0 = Input(shape=(128, 4, 1), name='EMB1'    )
        input1 = Input(shape=(16, 16, 1), name='EMB2'    )
        input2 = Input(shape=(8, 16, 1),  name='EMB3'    )
        input3 = Input(shape=(4, 4, 1),   name='TileBar0')
        input4 = Input(shape=(4, 4, 1),   name='TileBar1')
        input5 = Input(shape=(2, 4, 1),   name='TileBar2')
        inputs = [input0, input1, input2, input3, input4, input5]
        
        # Rescale our EMB images, and pass them through convolutions.
        EMB = ImageScaleBlock(input_shape,normalization=True, name_prefix='scaled_input_')([input0,input1,input2])
        if(augmentation): EMB = RandomFlip(name='aug_reflect')(EMB)
        if(normalization): EMB = NormalizationBlock(axes=[1,2,3])(EMB)
            
        EMB = ZeroPadding2D((3,3))(EMB)
        EMB = Conv2D(32, kernel, activation='relu')(EMB)
        EMB = MaxPooling2D(pool_size=(2,2))(EMB)
        EMB = Flatten()(EMB)
        
        # For TileBar, just get some simple info.
        # Using ImageScaleBlock with final size of (1,1) will give us a list of integrals of the images.
        TiB = ImageScaleBlock((1,1),normalization=True, name_prefix='scaled_input_TiB_')([input3,input4,input5])
        TiB = Flatten()(TiB)
        if(normalization): TiB = NormalizationBlock(axes=[1])(TiB)
        
        X = Concatenate(axis=1)([EMB,TiB])
        X = Dense(128, activation='relu')(X)
        if(dropout > 0.): X = Dropout(dropout)(X)
        X = Dense(64, activation='relu')(X)
        if(dropout > 0.): X = Dropout(dropout)(X)
        output = Dense(2, activation='softmax')(X)
        
        model = Model(inputs=inputs, outputs=output)
        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model


