import os, pathlib
import pandas as pd
from tensorflow.keras import Model # not sure if really needed?
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset, Options
import tensorflow as tf

# Given some input, indices, and a model filename,
# train an instance of the given model,
# or load the model if it already exists.
# One can pass data as numpy arrays (x_train, y_train, x_valid, y_valid)
def TrainNetwork(model, 
                 modelfile,
                 data_train=None,
                 data_valid=None,
                 x_train=None, y_train=None, 
                 x_valid=None, y_valid=None, 
                 callbacks = [],
                 sample_weight = None,
                 epochs=20, batch_size=200, verbose=1, 
                 overwriteModel=False, finishTraining=False,
                 custom_objects = None):
    '''
    Given some input, indices and a model filename, train an instance of the given model,
    or load the model if it already exists.
    
    One can pass data as numpy arrays (x_train, y_train, x_valid, y_valid) or as a tensorflow.data.Dataset (data_train, data_valid).
    If the data_train and data_valid arguments are given, then the numpy array arguments (x_train, y_train etc.) will be ignored.
    '''

    # Whether to use tf.data or numpy arrays.
    if(data_train is not None): use_tf_data = True
    else: use_tf_data = False
    
    # If no custom_objects provided, assume model type is our custom class, that packages the custom_objects with the model.
    if(custom_objects is None): model, custom_objects = model.model(), model.custom_objects
        
    # make the directory in which we save the model file
    model_dir = '/'.join(modelfile.split('/')[:-1])
    try: os.makedirs(model_dir)
    except: pass
        
    # Check if the model exists -- and load it if not overwriting.
#     if('.h5' or '.tf' in modelfile): history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'
#     else: history_filename = modelfile + '.csv' # if using .tf format, there won't be a file extension on the string at all.
    history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'    
    
    initial_epoch = 0
    
    do_training = True    
    if(pathlib.Path(modelfile).exists() and not overwriteModel):
        
        # If loading a model with custom layers, we need to pass a dictionary of custom layers to load_model.
        # This is very annoying -- it can be skipped if *not* saving to HDF5, but HDF5 is a convenient format
        # in that it gives us a single file, and we should support the option anyway.
        try: 
            model = load_model(modelfile, custom_objects=custom_objects)
            print('Successfully loaded model at {}'.format(modelfile))
             # Now we want to figure out for how many epochs the loaded model was already trained,
            # so that it's trained, in total, for the requested number of epochs.
            # keras models don't seem to hold on to an epoch attribute for whatever reason,
            # so we will figure out the current epoch based on CSVLogger output if it exists.
            if(pathlib.Path(history_filename).exists()):            
                with open(history_filename) as f:
                    for i,l in enumerate(f):
                        pass
                    initial_epoch = i
            if(not finishTraining): do_training = False
    
        except:
            print('Warning: Found modelfile {}, but could not load the model from it.'.format(modelfile))
            if(overwriteModel): 
                print('Will retrain and save to this file.')
                pass
            else: 
                print('Aborting to avoid overwriting file {}.'.format(modelfile))
                assert(False)
               

    # Train the model if we've specified "finishTraining", or if we don't even
    # have a model yet. Setting finishTraining=False lets one immediately skip
    # to evaluating the model, which is especially helpful if EarlyStopping was used
    # and the final model didn't reach the specified last epoch.
    if(do_training):
        
#         # For TensorFlow 2.4+, we need to wrap our data in a TF data object
#         # to avoid some warnings about "sharding" in certain cases.
#         # See https://stackoverflow.com/a/65344405/14765225
#         train_data = Dataset.from_tensor_slices((x_train, y_train))
#         valid_data = Dataset.from_tensor_slices((x_valid, y_valid))
#         train_data = train_data.batch(batch_size)
#         valid_data = valid_data.batch(batch_size)
        
#         # Disable AutoShard.
#         options = Options()
#         options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#         train_data = train_data.with_options(options)
#         valid_data = valid_data.with_options(options)

        if(sample_weight == None): sample_weight=(None,None)
            
        if(not use_tf_data):
            
            history = model.fit(
                x_train, y_train,
                validation_data=(
                    x_valid,
                    y_valid,
                    sample_weight[1]
                ),
                epochs=epochs,
                initial_epoch=initial_epoch,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
                sample_weight=sample_weight[0]
            )
            
        else:
            
            #data_train = data_train.batch(batch_size)
            #data_valid = data_valid.batch(batch_size)
            
            # Disable autosharding. TODO: Is this necessary?
            #options = Options()
            #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            #data_train = data_train.with_options(options)
            #data_valid = data_valid.with_options(options)
            
            history = model.fit(
                data_train,
                validation_data=data_valid,
                epochs=epochs,
                initial_epoch=initial_epoch,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
                sample_weight=sample_weight[0]
            )
            
    saveModel = True
    if(initial_epoch == epochs or not finishTraining): saveModel = False
    if(saveModel):
        print('  Saving model to {}.'.format(modelfile))
        model.save(modelfile)
        
    # Now get the history from the log file, if it exists.
    # This is a better method than using the results of model.fit(),
    # since this will give us the whole history (not just whatever
    # was fitted right now). However, it relies on us having passed
    # a CSVLogger as one of our callbacks, which we normally do
    # but might not do in some specific circumstances.
    
    # fallback
    try: 
        history = history.history
    except: 
        history = {}
        pass
        
    if(pathlib.Path(history_filename).exists()):
        df = pd.read_csv(history_filename)
        history = {}
        for key in df.keys():
            history[key] = df[key].to_numpy()
        
    else:
        print('Warning: No log file found for model {}.'.format(modelfile))
        print('This may result in an empty/incomplete history being returned.')
        print('Please provide a CSVLogger callback to prevent this in the future.')

    return model, history