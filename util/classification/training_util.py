import os, pathlib
import pandas as pd
from tensorflow.keras import Model # not sure if really needed?
from tensorflow.keras.models import load_model

# Given some input, indices, and a model filename,
# train an instance of the given model.
def TrainNetwork(model, modelfile, x_train, y_train, x_valid, y_valid, callbacks = [], epochs=20, batch_size=200, verbose=1, overwriteModel=False):
    
    # make the directory in which we save the model file
    model_dir = '/'.join(modelfile.split('/')[:-1])
    try: os.makedirs(model_dir)
    except: pass
        
    # Check if the model exists -- and load it if not overwriting.
    history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'
    initial_epoch = 0
    if(pathlib.Path(modelfile).exists() and not overwriteModel):
        model = load_model(modelfile)
        
        # Now we want to figure out for how many epochs the loaded model was already trained,
        # so that it's trained, in total, for the requested number of epochs.
        # keras models don't seem to hold on to an epoch attribute for whatever reason,
        # so we will figure out the current epoch based on CSVLogger output if it exists.
        if(pathlib.Path(history_filename).exists()):            
            with open(history_filename) as f:
                for i,l in enumerate(f):
                    pass
                initial_epoch = i
                
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
    if(initial_epoch == epochs): saveModel = False
    if(saveModel):
        print('  Saving model to {}.'.format(modelfile))
        model.save(modelfile)
        
    # Now get the history from the log file, if it exists.
    # This is a better method than using the results of model.fit(),
    # since this will give us the whole history (not just whatever
    # was fitted right now). However, it relies on us having passed
    # a CSVLogger as one of our callbacks, which we normally do
    # but might not do in some specific circumstances.
    history = history.history # fallback
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