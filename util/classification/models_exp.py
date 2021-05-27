# More experimental models / less organized code that we don't want cluttering
# up our models.py file.

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Add, Concatenate, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
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

# Based on Max's best-performing model from his "Large CNN" notebook.
# The best results were previously achieved using dropout=0.2 .
class exp_merged_model():
    def __init__(self, lr=5e-5, dropout=-1., augmentation=True, normalization=True):
        self.lr = lr
        self.dropout = dropout
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock
        }
        
    def model(self):
        lr = self.lr
        dropout = self.dropout
        augmentation = self.augmentation
        normalization = self.normalization
        
        # Gather inputs. We will reshape TB2.
        EMB1 = Input(shape=(128,4,1), name='EMB1')
        EMB2 = Input(shape=(16,16,1), name='EMB2')
        EMB3 = Input(shape=(8,16,1), name='EMB3')
        TB0 = Input(shape=(4,4,1), name='TileBar0')
        TB1 = Input(shape=(4,4,1), name='TileBar1')
        TB2 = Input(shape=(2,4,1), name='TileBar2')
        
        input1 = EMB1
        input2 = ImageScaleBlock((16,16),normalization=True, name_prefix='emb_stack')([EMB2, EMB3]) # merge EMB2 and EMB3
        input3 = ImageScaleBlock((4,4),normalization=True, name_prefix='tiles_stack')([TB0, TB1, TB2]) # merge TileBar

        # From here on out, just follow Max's code.
        
        # EMB1 image (convolutional)
        x1 = Conv2D(32, (3, 3), padding='same', name='emb1_conv2d_1')(input1)
        x1 = Activation('relu')(x1)
        # x1 = Dropout(dropout)(x1)
        x1 = Conv2D(32, (3, 3), padding='same', name='emb1_conv2d_2')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 1), padding='same', name='emb1_maxpool_3')(x1)
        x1 = Conv2D(64, (3, 3), padding='same', name='emb1_conv2d_3')(x1)
        x1 = Activation('relu')(x1)
        #x1 = Dropout(dropout)(x1)
        x1 = Conv2D(64, (3, 3), padding='same', name='emb1_conv2d_4')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 1), padding='same', name='emb1_maxpool_5')(x1)
        x1 = Conv2D(128, (2, 2), padding='same', name='emb1_conv2d_6')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(128, (2, 2), padding='same', name='emb1_conv2d_7')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 1), padding='same', name='emb1_maxpool_8')(x1)
        x1 = Dropout(dropout, name='emb1_dropout_4')(x1)
        x1 = Flatten(name='emb1_flatten_9')(x1)
        x1 = Dense(128, activation='relu', name='emb1_dense_9')(x1)

        # EMB23 image (convolutional)
        x2 = Conv2D(32, (1, 1), padding='same', name='emb23_conv1d_1')(input2)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(dropout)(x2)
        x2 = Conv2D(64, (2, 2), padding='same', name='emb23_conv2d_2')(x2)
        # x2 = Dropout(dropout)(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='emb23_maxpool_3')(x2)
        x2 = Conv2D(128, (2, 2), padding='same', name='emb23_conv2d_4')(x2)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(dropout)(x2)
        x2 = Conv2D(128, (2, 2), padding='same', name='emb23_conv2d_5')(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='emb23_maxpool_6')(x2)
        x2 = Dropout(dropout, name='emb23_dropout_4')(x2)
        x2 = Flatten(name='emb23_flatten_7')(x2)
        x2 = Dense(128, activation='relu', name='emb23_dense_8')(x2)

        # tiles image (convolutional)
        x3 = Conv2D(32, (1, 1), padding='same', name='tiles_conv1d_1')(input3)
        x3 = Activation('relu')(x3)
        # x3 = Dropout(dropout)(x3)
        x3 = Conv2D(64, (2, 2), padding='same', name='tiles_conv2d_2')(x3)
        x3 = Activation('relu')(x3)
        # x3 = Dropout(dropout)(x3)
        x3 = Conv2D(128, (2, 2), padding='same', name='tiles_conv2d_3')(x3)
        x3 = Activation('relu')(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='tiles_maxpool_4')(x3)
        x3 = Dropout(dropout, name='tiles_dropout_4')(x3)
        x3 = Flatten(name='tiles_flatten_5')(x3)
        x3 = Dense(128, activation='relu', name='tiles_dense_6')(x3)

        # concatenate outputs from the two networks above
        x = Concatenate(axis=1, name='concatenate')([x1, x2, x3])
        #x = concatenate([x1, x2, x3], name='concatenate') 
        x = Dropout(dropout, name='concate_dropout_5')(x)
        x = Dense(64, name='concated_dense_1')(x)    
        x = Activation('relu')(x)
        x = Dropout(dropout, name='dense_dropout_6')(x)

        # final output
        output = Dense(2, activation='softmax', name='dense_output')(x)
        # output = 5*tf.math.tanh(x)   # 0 to +5 range

        model = Model(inputs = [EMB1, EMB2, EMB3, TB0, TB1, TB2], outputs = [output])
        # compile model
        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model