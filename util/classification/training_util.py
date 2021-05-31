import os, pathlib
import pandas as pd
from tensorflow.keras import Model # not sure if really needed?
from tensorflow.keras.models import load_model

# Given some input, indices, and a model filename,
# train an instance of the given model,
# or load the model if it already exists.
def TrainNetwork(model, 
                 modelfile, 
                 x_train, y_train, 
                 x_valid, y_valid, 
                 callbacks = [], 
                 epochs=20, batch_size=200, verbose=1, 
                 overwriteModel=False, finishTraining=True,
                 custom_objects = {}):
    
    model, custom_objects = model.model(), model.custom_objects
    
    # make the directory in which we save the model file
    model_dir = '/'.join(modelfile.split('/')[:-1])
    try: os.makedirs(model_dir)
    except: pass
        
    # Check if the model exists -- and load it if not overwriting.
    # TODO: Maybe consider a neater way to deal with different file extensions (or lack thereof)?
    history_filename = 0
    if('.h5' in modelfile): history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'
    else: history_filename = modelfile + '.csv' # if using .tf format, there won't be a file extension on the string at all.
    initial_epoch = 0
    if(pathlib.Path(modelfile).exists() and not overwriteModel):
        
        # If loading a model with custom layers, we need to pass a dictionary of custom layers to load_model.
        # This is very annoying -- it can be skipped if *not* saving to HDF5, but HDF5 is a convenient format
        # in that it gives us a single file, and we should support the option anyway.
        
        model = load_model(modelfile, custom_objects=custom_objects)
        
        # Now we want to figure out for how many epochs the loaded model was already trained,
        # so that it's trained, in total, for the requested number of epochs.
        # keras models don't seem to hold on to an epoch attribute for whatever reason,
        # so we will figure out the current epoch based on CSVLogger output if it exists.
        if(pathlib.Path(history_filename).exists()):            
            with open(history_filename) as f:
                for i,l in enumerate(f):
                    pass
                initial_epoch = i
         
    history = 0
    
    # Train the model if we've specified "finishTraining", or if we don't even
    # have a model yet. Setting finishTraining=False lets one immediately skip
    # to evaluating the model, which is especially helpful if EarlyStopping was used
    # and the final model didn't reach the specified last epoch.
    if(finishTraining or not pathlib.Path(modelfile).exists()):
        history = model.fit(
            x_train, y_train,
            validation_data=(
                x_valid,
                y_valid
            ),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
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
        print('Warning: No log file found for model {}.'.format())
        print('This may result in an empty/incomplete history being returned.')
        print('Please provide a CSVLogger callback to prevent this in the future.')

    return model, history