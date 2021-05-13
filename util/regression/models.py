import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
#from keras.wrappers.scikit_learn import KerasRegressor
from string import ascii_lowercase

# Custom layers.
from util.keras.layers import *

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
    
    