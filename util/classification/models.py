import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Add, Concatenate, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from string import ascii_lowercase

# Custom layers.
from util.keras.layers import *

# Our baseline, fully-connected neural network for classification.
# Operates on a single vector (e.g. flattened image from one calorimeter layer).
# Optionally uses dropouts between layers.
class baseline_nn_model(): 
    def __init__(self, strategy, number_pixels, lr=5e-5, dropout=-1, normalization=True, input_name='input'):
        self.strategy = strategy
        self.dropout = dropout
        self.normalization = normalization
        self.number_pixels = number_pixels
        self.lr = lr
        self.input_name = input_name
        self.custom_objects = {
            'NormalizationBlock':NormalizationBlock
        } 
        
    # create model
    def model(self):
        dropout = self.dropout
        normalization = self.normalization
        used_pixels = self.number_pixels
        strategy = self.strategy
        lr = self.lr
        input_name = self.input_name
        #with strategy.scope():                
        X_in = Input((used_pixels),name=self.input_name)
        X = X_in

        # If requested, normalize the flattened input images to integrate to one.
        # This removes any information on the energy scale that may be in the input images.
        if(normalization): X = NormalizationBlock(axes=[1])(X)

        X = Dense(used_pixels, kernel_initializer='normal',activation='relu')(X)
        if(dropout > 0.): X = Dropout(dropout)(X)

        X = Dense(used_pixels, activation='relu')(X)
        if(dropout > 0.): X = Dropout(dropout)(X)

        X = Dense(int(used_pixels/2), activation='relu')(X)
        if(dropout > 0.): X = Dropout(dropout)(X)

        X = Dense(2, kernel_initializer='normal', activation='softmax')(X)

        # compile the model
        model = Model(inputs=X_in, outputs=X, name='base_nn')
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

# A "simple" convolutional neural network (CNN). Uses a single calo image.
class baseline_cnn_model():
    def __init__(self, input_shape, f, pool, lr=5e-5, augmentation=True, normalization=True):
        self.lr = lr
        self.input_shape = tuple(list(input_shape) + [1])
        self.f = f
        self.pool = pool
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'NormalizationBlock': NormalizationBlock
        }
        
    def model(self):
        lr = self.lr
        input_shape = self.input_shape
        f = self.f
        pool = self.pool
        augmentation = self.augmentation
        normalization = self.normalization
        
        X_in = Input(input_shape,name='input')
        
        # Augmentation: randomly flip the image during training
        if(augmentation): X = RandomFlip(name='aug_reflect')(X_in)
        else: X = X_in
            
        # Normalization: Rescale the image to have an integral of 1
        if(normalization): X = NormalizationBlock(axes=[1,2])(X)
                
        X = Conv2D(32, f, name='conv1', activation='relu')(X)            
        #temp_pool = (2,2) #TODO: Including this pooling caused model to fail to compile. Did this work in original CNN notebook w/ original parameter choices?
        #X = MaxPooling2D(temp_pool)(X)            
        X = Conv2D(16, pool, activation='relu')(X)
        X = MaxPooling2D(pool)(X)
        X = Flatten()(X)
        X = Dense(128,activation='relu')(X)
        X = Dense(50,activation='relu')(X)
        X = Dense(2, kernel_initializer='normal', activation='softmax')(X)
        model = Model(inputs=X_in, outputs=X, name='SL_CNN')
        # compile model
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

# A model that uses all 3 EMB layers (as separate, 1-channel images).
# As a result, the input shapes are hard-coded.
class emb_cnn_model():
    def __init__(self, lr=5e-5, augmentation=True, normalization=True):
        self.lr = lr
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'NormalizationBlock': NormalizationBlock
        }
        
    def model(self):
        lr = self.lr
        augmentation = self.augmentation
        normalization = self.normalization
        
        # EMB1 image (convolutional)
        input0 = Input(shape=(128, 4, 1), name='EMB1')
        if(augmentation): X0 = RandomFlip(name='aug_reflect_0')(input0)            
        else: X0 = input0
            
        if(normalization): X0 = NormalizationBlock(axes=[1,2])(X0)
        X0 = Conv2D(32, (4, 2), activation='relu')(X0)
        X0 = MaxPooling2D(pool_size=(2, 2))(X0)
        X0 = Dropout(0.2)(X0)
        X0 = Flatten()(X0)
        X0 = Dense(128, activation='relu')(X0)

        # EMB2 image (convolutional)
        input1 = Input(shape=(16, 16, 1), name='EMB2')
        if(augmentation): X1 = RandomFlip(name='aug_reflect_1')(input1)
        else: X1 = input1
            
        if(normalization): X1 = NormalizationBlock(axes=[1,2])(X1)
        X1 = Conv2D(32, (4, 4), activation='relu')(X1)
        X1 = MaxPooling2D(pool_size=(2, 2))(X1)
        X1 = Dropout(0.2)(X1)
        X1 = Flatten()(X1)
        X1 = Dense(128, activation='relu')(X1)
    
        # EMB3 image (convolutional)
        input2 = Input(shape=(8, 16, 1), name='EMB3')
        if(augmentation): X2 = RandomFlip(name='aug_reflect_2')(input2)
        else: X2 = input2
            
        if(normalization): X2 = NormalizationBlock(axes=[1,2])(X2)
        X2 = Conv2D(32, (2, 4), activation='relu')(X2)
        X2 = MaxPooling2D(pool_size=(2, 2))(X2)
        X2 = Dropout(0.2)(X2)
        X2 = Flatten()(X2)
        X2 = Dense(128, activation='relu')(X2)

        # concatenate outputs from the three networks above
        X = Concatenate(axis=1)([X0, X1, X2]) # remember that axis=0 is batch!
        X = Dense(50, activation='relu')(X)    

        # final output
        X = Dense(2, activation='softmax')(X)
        model = Model(inputs = [input0, input1, input2], outputs=X)
    
        # compile model
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    
# A model that uses all 6 calo layers (as separate, 1-channel images).
# As a result, the input shapes are hard-coded.
class all_cnn_model():
    def __init__(self, lr=5e-5, augmentation=True, normalization=True):
        self.lr = lr
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'NormalizationBlock': NormalizationBlock
        }
        
    def model(self):
        lr = self.lr
        augmentation = self.augmentation
        normalization = self.normalization
        
        # EMB1 image (convolutional)
        input0 = Input(shape=(128, 4, 1), name='EMB1')
        if(augmentation): X0 = RandomFlip(name='aug_reflect_0')(input0)
        else: X0 = input0
        
        if(normalization): X0 = NormalizationBlock(axes=[1,2])(X0)
        X0 = Conv2D(32, (4, 2), activation='relu')(X0)
        X0 = MaxPooling2D(pool_size=(2, 2))(X0)
        X0 = Dropout(0.2)(X0)
        X0 = Flatten()(X0)
        X0 = Dense(128, activation='relu')(X0)

        # EMB2 image (convolutional)
        input1 = Input(shape=(16, 16, 1), name='EMB2')
        if(augmentation): X1 = RandomFlip(name='aug_reflect_1')(input1)
        else: X1 = input1

        if(normalization): X1 = NormalizationBlock(axes=[1,2])(X1)
        X1 = Conv2D(32, (4, 4), activation='relu')(X1)
        X1 = MaxPooling2D(pool_size=(2, 2))(X1)
        X1 = Dropout(0.2)(X1)
        X1 = Flatten()(X1)
        X1 = Dense(128, activation='relu')(X1)
    
        # EMB3 image (convolutional)
        input2 = Input(shape=(8, 16, 1), name='EMB3')
        if(augmentation): X2 = RandomFlip(name='aug_reflect_2')(input2)
        else: X2 = input2
        
        if(normalization): X2 = NormalizationBlock(axes=[1,2])(X2)
        X2 = Conv2D(32, (2, 4), activation='relu')(X2)
        X2 = MaxPooling2D(pool_size=(2, 2))(X2)
        X2 = Dropout(0.2)(X2)
        X2 = Flatten()(X2)
        X2 = Dense(128, activation='relu')(X2)
        
        #TileBar0 image (convolutional)
        input3 = Input(shape=(4,4,1), name='TileBar0')
        if(augmentation): X3 = RandomFlip(name='aug_reflect_3')(input3)
        else: X3 = input3
        
        if(normalization): X3 = NormalizationBlock(axes=[1,2])(X3)
        X3 = Conv2D(32, (2,2), activation='relu')(X3)
        X3 = MaxPooling2D(pool_size=(2,2))(X3)
        X3 = Dropout(0.2)(X3)
        X3 = Flatten()(X3)
        X3 = Dense(128, activation='relu')(X3)

        #TileBar1 image (convolutional)
        input4 = Input(shape=(4,4,1), name='TileBar1')
        if(augmentation): X4 = RandomFlip(name='aug_reflect_4')(input4)
        else: X4 = input4
        
        if(normalization): X4 = NormalizationBlock(axes=[1,2])(X4)
        X4 = Conv2D(32, (2,2), activation='relu')(X4)
        X4 = MaxPooling2D(pool_size=(2,2))(X4)
        X4 = Dropout(0.2)(X4)
        X4 = Flatten()(X4)
        X4 = Dense(128, activation='relu')(X4)
        
        #TileBar2 image (convolutional)
        input5 = Input(shape=(2,4,1), name='TileBar2')
        if(augmentation): X5 = RandomFlip(name='aug_reflect_5')(input5)
        else: X5 = input5
        
        if(normalization): X5 = NormalizationBlock(axes=[1,2])(X5)
        X5 = Conv2D(32, (2,2), activation='relu')(X5)
        X5 = MaxPooling2D(pool_size=(1,2))(X5)
        X5 = Dropout(0.2)(X5)
        X5 = Flatten()(X5)
        X5 = Dense(128, activation='relu')(X5)
        
        # concatenate outputs from the three networks above
        X = Concatenate(axis=1)([X0, X1, X2, X3, X4, X5]) # remember that axis=0 is batch!
        X = Dense(50, activation='relu')(X)    

        # final output
        X = Dense(2, activation='softmax')(X)
        model = Model(inputs = [input0, input1, input2, input3, input4, input5], outputs=X)
    
        # compile model
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    
# A simple CNN, that uses 6-channel images.
# Each set of images is appropriately rescaled so that their dimensions match.
class merged_cnn_model():
    def __init__(self, input_shape, lr=5e-5, dropout=-1., augmentation=True, normalization=True):
        self.lr = lr
        self.input_shape = input_shape
        self.dropout = dropout
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock,
            'NormalizationBlock':NormalizationBlock
        } 
        
    def model(self):
        input_shape = self.input_shape
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
        
        # Rescale all our images.
        X = ImageScaleBlock(input_shape,normalization=True, name_prefix='scaled_input_')(inputs)
        if(augmentation): X = RandomFlip(name='aug_reflect')(X)
        if(normalization): X = NormalizationBlock(axes=[1,2,3])(X)

        X = ZeroPadding2D((3,3))(X)
                
        X = Conv2D(32, (int(input_shape[0]/4), int(input_shape[1]/4)), activation='relu')(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)
        if(dropout > 0.): X = Dropout(dropout)(X)
        X = Flatten()(X)
        X = Dense(128, activation='relu')(X)
        output = Dense(2, activation='softmax')(X)
        
        model = Model(inputs=inputs, outputs=output)
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

# A CNN that uses two 3-channel images,
# one corresponding with EMB and the other with TileBar.
class merged_cnn_2p_model():
    def __init__(self, input_shape1, input_shape2, lr=5e-5, dropout=-1., augmentation=True, normalization=True):
        self.lr = lr
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2
        self.dropout = dropout
        self.augmentation = augmentation
        self.normalization = normalization
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock,
            'NormalizationBlock':NormalizationBlock
        } 

    def model(self):
        input_shapes = [self.input_shape1, self.input_shape2]
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
        
        inputs_EMB = [input0, input1, input2]
        inputs_TileBar = [input3, input4, input5]
        
        Xs = []
        for i,input_list in enumerate([inputs_EMB, inputs_TileBar]):
            input_shape = input_shapes[i]
            X = ImageScaleBlock(input_shape,normalization=True, name_prefix='scaled_input_')(inputs)
            if(augmentation): X = RandomFlip(name='aug_reflect_{}'.format(i))(X)
            if(normalization): X = NormalizationBlock(axes=[1,2,3])(X)
            X = ZeroPadding2D((3,3))(X)
            X = Conv2D(32, (int(input_shape[0]/4), int(input_shape[1]/4)), activation='relu')(X)
            X = MaxPooling2D(pool_size=(2, 2))(X)
            if(dropout > 0.): X = Dropout(dropout)(X)
            X = Flatten()(X)
            Xs.append(X)
            
        # concatenate results
        X = Concatenate(axis=1)(Xs)
        X = Dense(128, activation='relu')(X)
        output = Dense(2, activation='softmax')(X)
        
        model = Model(inputs=inputs, outputs=output)
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    
# A simple implementation of ResNet.
# As input, this takes multiple images, which may be of different sizes,
# and they are all rescaled to a user-specified size.
class resnet(): 
    def __init__(self, filter_sets, f_vals, s_vals, i_vals, channels=6, lr=5e-5, input_shape = (128,16), augmentation=True, normalization=True, classes=2):
        self.filter_sets = filter_sets
        self.f_vals = f_vals
        self.s_vals = s_vals
        self.i_vals = i_vals
        self.channels = channels
        self.input_shape = input_shape
        self.augmentation = augmentation
        self.normalization = normalization
        self.lr = lr
        self.classes = classes
        self.custom_objects = {
            'ImageScaleBlock':ImageScaleBlock,
            'ConvolutionBlock':ConvolutionBlock,
            'IdentityBlock':IdentityBlock,
            'NormalizationBlock':NormalizationBlock
        }
    
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
        classes = self.classes
        
        # Input images -- one for each channel, each channel's dimensions may be different.
        inputs = [Input((None,None,1),name='input'+str(i)) for i in range(channels)]

        # Rescale all the input images, so that their dimensions now match.
        X = ImageScaleBlock(input_shape,normalization=True,name_prefix='scaled_input_')(inputs)
        if(normalization): X = NormalizationBlock(axes=[1,2,3])(X)

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

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

        # Create model object.
        model = Model(inputs=inputs, outputs=X, name='ResNet')

        # Compile the model
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

# A simple combination network -- in practice we may use this to
# "combine" classification scores from single calo-layer networks.
class simple_combine_model(): 
    def __init__(self, strategy, lr=5e-5, n_input=6):
        self.strategy = strategy
        self.n_input = n_input
        self.lr = lr
        self.custom_objects = {}
    
    # create model
    def model(self):
        n_input = self.n_input
        strategy = self.strategy
        lr = self.lr
        with strategy.scope():
            model = Sequential()
            model.add(Dense(n_input, input_dim=n_input, kernel_initializer='normal', activation='relu'))
            model.add(Dense(4, activation='relu')) # TODO: Change width from 4 to n_input?
            model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
            # compile model
            optimizer = Adam(learning_rate=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

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
        optimizer = Adam(learning_rate=lr)
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
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model