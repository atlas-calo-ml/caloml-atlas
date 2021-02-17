import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor

# Custom layers.
from layers import *

# A simple, fully-connected network architecture.
# Inputs correspond with the pixels of all the images,
# plus the reco energy (possibly transformed by a logarithm),
# and the eta of the cluster. (This is our baseline model).
def baseline_nn_All_model(strategy, lr=1e-4, decay=1e-6, dropout=-1.):
    number_pixels = 512 + 256 + 128 + 16 + 16 + 8
    # create model
    def mod():    
        with strategy.scope():    
            model = Sequential()
            used_pixels = number_pixels + 2
            model.add(Dense(used_pixels, input_dim=used_pixels, kernel_initializer='normal', activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            
            model.add(Dense(used_pixels, activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))

            model.add(Dense(int(used_pixels/2), activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))

            model.add(Dense(units=1, kernel_initializer='normal', activation='linear'))
            
            opt = Adam(lr=lr, decay=decay)
            model.compile(optimizer=opt, loss='mse',metrics=['mae','mse'])
        return model
    return mod


# An implementation of ResNet.
# Inputs correspond with calorimeter images, as well as the reco energy and and eta.
def resnet(strategy, channels=6, lr=5e-5):
    # create model
    def mod(input_shape=(128,16)):
        with strategy.scope():
            
            # First, the real ResNet portion of the network -- using the images.
            
            # Input images -- one for each channel, each channel's dimensions may be different.
            inputs = [Input((None,None,1),name='input'+str(i)) for i in range(channels)]
            
            # Additional inputs -- not used for ResNet portion, but introduced near the end.
            energy_input = Input(shape =(1,), name='energy')
            eta_input    = Input(shape =(1,), name='eta')
            
            # Rescale all the input images, so that their dimensions now match.
            scaled_inputs = [tf.image.resize(x,input_shape,name='scaled_input'+str(i)) for i,x in enumerate(inputs)]
            
            # Now "stack" the images along the channels dimension.
            X = tf.concat(values=scaled_inputs, axis=3, name='concat')
            #print('In:',X.shape)
            
            X = ZeroPadding2D((3,3))(X)
            #print('S0:',X.shape)
            
            # Stage 1
            X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name='bn_conv1')(X)
            X = Activation('relu')(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)
            #print('S1:',X.shape)
            
            # Stage 2
            filters = [64, 64, 256]
            f = 3
            stage = 2
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=1)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S2:',X.shape)
            
            # Stage 3
            filters = [128, 128, 512]
            f = 3
            stage = 3
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            #print('S3:',X.shape)

            # Stage 4
            filters = [256, 256, 1024]
            f = 3
            stage = 4
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            X = identity_block(X, f, filters, stage=stage, block='e')
            X = identity_block(X, f, filters, stage=stage, block='f')
            #print('S4:',X.shape)

            # Stage 5
            filters = [512, 512, 2048]
            f = 3
            stage = 5
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S5:',X.shape)

            # AVGPOOL
            pool_size = (2,2)
            if(X.shape[1] == 1):   pool_size = (1,2)
            elif(X.shape[2] == 1): pool_size = (2,1)
            X = AveragePooling2D(pool_size=pool_size, name="avg_pool")(X)
            #print('S6:',X.shape)

            # Flatten the ResNet output.
            X = Flatten()(X)
            #print('S7:',X.shape)
            
            # Now add in the input energy and eta (energy might've been rescaled, e.g. by a logarithm).
            # See https://github.com/tensorflow/tensorflow/issues/30355#issuecomment-553340170
            tensor_list = [X,energy_input,eta_input]
            X = Concatenate(axis=1)(tensor_list[:])
            #X = concatenate([tensor_list])
            #print('S8:',X.shape)
            
            X = Dense(units=1, activation='linear', name='output', kernel_initializer='normal')(X)
    
            # Create model object.
            input_list = inputs + [energy_input, eta_input]
            model = Model(inputs=input_list, outputs=X, name='ResNet50')
        
            # Compile the model
            optimizer = Adam(lr=lr)
            model.compile(optimizer=optimizer, loss='mse',metrics=['mae','mse'])
        return model
    return mod
