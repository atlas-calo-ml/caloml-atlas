import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input, Flatten
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
#from keras.wrappers.scikit_learn import KerasRegressor
from string import ascii_lowercase

# Custom layers.
from util.keras.layers import *
#from util import ml_util as mu

# A simple, fully-connected network architecture.
# Inputs correspond with the pixels of all the images,
# plus the reco energy (possibly transformed by a logarithm),
# and the eta of the cluster. (This is our baseline model).
class baseline_nn_model(): 
    def __init__(self, strategy, lr=5e-5, decay=1e-6, dropout=-1):
        self.strategy = strategy
        self.dropout = dropout
        self.decay = decay
        self.lr = lr
        self.custom_objects = {}
    
    # create model
    def model(self):
        dropout = self.dropout
        decay = self.decay
        strategy = self.strategy
        lr = self.lr
        number_pixels = 512 + 256 + 128 + 16 + 16 + 8
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

# A simple, fully-connected network architecture.
# Inputs correspond to the reco energy, eta, as well as a vector
# encoding the percentage of energy deposited in each calorimeter layer.
class simple_dnn():
    def __init__(self, strategy, lr=5e-5, decay=1e-6, dropout=-1):
        self.strategy = strategy
        self.dropout = dropout
        self.decay = decay
        self.lr = lr
        self.custom_objects = {}
        
    # create model
    def model(self):
        dropout = self.dropout
        decay = self.decay
        strategy = self.strategy
        lr = self.lr
        number_pixels = 512 + 256 + 128 + 16 + 16 + 8
        with strategy.scope():    
            energy_input = Input(shape=(1,),name='energy')
            eta_input    = Input(shape =(1,), name='eta')
            depth_input = Input(shape=(6,),name='depth')
            
            input_list = [energy_input, eta_input, depth_input]
            
            X = tf.concat(values = input_list,axis=1,name='concat')
            X = Dense(units=8, activation='relu',name='Dense1')(X)
            X = Dense(units=8, activation='relu',name='Dense2')(X)
            X = Dense(units=8, activation='relu',name='Dense3')(X)
            X = Dense(units=1, kernel_initializer='normal', activation='linear')(X)
            
            optimizer = Adam(lr=lr, decay=decay)
            model = Model(inputs=input_list, outputs=X, name='Simple')
            model.compile(optimizer=optimizer, loss='mse',metrics=['mae','mse'])
        return model    
    
class resnet():
    def __init__(self, lr, decay, filter_sets, f_vals, s_vals, i_vals, channels=6, input_shape=(128,16), augmentation=True, normalization=True):

        self.decay = decay
        self.lr = lr
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock,
            'ConvolutionBlock':ConvolutionBlock,
            'IdentityBlock':IdentityBlock
        }
        
        self.filter_sets = filter_sets
        self.f_vals = f_vals
        self.s_vals = s_vals
        self.i_vals = i_vals
        
        self.channels = channels
        self.input_shape = input_shape
        self.augmentation = augmentation
        self.normalization = normalization
        
    # create model
    def model(self):
        filter_sets = self.filter_sets
        f_vals = self.f_vals
        s_vals = self.s_vals
        i_vals = self.i_vals
        channels = self.channels
        input_shape = self.input_shape
        augmentation = self.augmentation
        normalization = self.normalization
        lr = self.lr
        decay = self.decay
        
        # Input images -- one for each channel, each channel's dimensions may be different.
        inputs = [Input((None,None,1),name='input_'+str(i)) for i in range(channels)]
        eta_input = Input((1),name='eta')

        # Rescale all the input images, so that their dimensions now match.
        # Note that we make sure to re-normalize the images so that we preserve their energies.
        X = ImageScaleBlock(input_shape,normalization=True,name_prefix='scaled_input_')(inputs)
        X = ZeroPadding2D((3,3))(X)

        # Data augmentation.
        # With channels combined, we can now flip images in (eta,phi, eta&phi).
        # Note that these flips will not require making any changes to
        # the other inputs (energy, abs(eta)), so the augmentation is
        # as simple as flipping the images using built-in functions.
        # These augmentation functions will only be active during training.
        if(augmentation): X = RandomFlip(name='aug_reflect')(X)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
       
        n = len(f_vals)
        for i in range(n):
            filters = filter_sets[i]
            f = f_vals[i]
            s = s_vals[i]
            ib = i_vals[i]
            stage = i + 1 # 1st stage is Conv2D etc. before ResNet blocks
            #X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=1)
            X = ConvolutionBlock(f=f, filters=filters, stage=stage, block='a', s=s, normalization=normalization)(X)
            
            for j in range(ib):
                #X = identity_block(X, f, filters, stage=stage, block=ascii_lowercase[j+1]) # will only cause naming issues if there are many id blocks
                X = IdentityBlock(f=f, filters=filters, stage=stage, block=ascii_lowercase[j+1],normalization=normalization)(X)

        # AVGPOOL
        pool_size = (2,2)
        if(X.shape[1] == 1):   pool_size = (1,2)
        elif(X.shape[2] == 1): pool_size = (2,1)
        X = AveragePooling2D(pool_size=pool_size, name="avg_pool")(X)
        
        # Transition into output: Add eta info, and use a simple DNN.
        X = Flatten()(X)
        tensor_list = [X,eta_input]
        X = Concatenate(axis=1)(tensor_list)
        units = X.shape[1]
        X = Dense(units=units, activation='relu',name='Dense1')(X)
        X = Dense(units=int(units/4), activation='relu',name='Dense2')(X)
        X = Dense(units=1, activation='linear', name='output', kernel_initializer='normal')(X)

        # Create model object.
        input_list = inputs + [eta_input]
        model = Model(inputs=input_list, outputs=X, name='ResNet')

        # Compile the model
        optimizer = Adam(lr=lr,decay=decay)
        model.compile(optimizer=optimizer, loss='mse',metrics=['mae','mse'])
        return model


        
        