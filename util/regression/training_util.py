import os, pickle, pathlib
import pandas as pd
import h5py as h5
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model

# Given sets of input a model filename,
# train a single model.
def TrainNetwork(model, 
                 modelfile, 
                 x_train, y_train, 
                 x_valid, y_valid, 
                 sample_weight, 
                 callbacks,
                 epochs=20, batch_size=200, verbose=1, 
                 overwriteModel=False, finishTraining=True):
    
    model, custom_objects = model.model, model.custom_objects

    
    # Set up our KerasRegressor wrapper.
    # I'm not 100% sure why we do this for our regressors (but not our classifiers),
    # but as we use this in the original training code I'll keep it for now.
    regressor = KerasRegressor(
        build_fn = model,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose
    )
    
    # Make the model directory if it does not already exist.
    model_dir = '/'.join(modelfile.split('/')[:-1])
    try: os.makedirs(model_dir)
    except: pass
    
    # Check if the model exists -- and load it if not overwriting.
    history_filename = 0
    if('.h5' in modelfile): history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'
    else: history_filename = modelfile + '.csv' # if using .tf format, there won't be a file extension on the string at all.
    initial_epoch = 0
    if(pathlib.Path(modelfile).exists() and not overwriteModel):
        regressor.model = load_model(modelfile, custom_objects=custom_objects)
        
        # Now we want to figure out for how many epochs the loaded model was already trained,
        # so that it's trained, in total, for the requested number of epochs.
        # keras models don't seem to hold on to an epoch attribute for whatever reason,
        # so we will figure out the current epoch based on CSVLogger output if it exists.
        if(pathlib.Path(history_filename).exists()):            
            with open(history_filename) as f:
                for i,l in enumerate(f):
                    pass
                initial_epoch = i # zero-indexing will take care of the 1st line, which has headers
        if(not finishTraining): initial_epoch = regressor.get_params()['epochs']
        regressor.set_params(initial_epoch=initial_epoch)
       
    history = 0
    # Train the model if we've specified "finishTraining", or if we don't even
    # have a model yet. Setting finishTraining=False lets one immediately skip
    # to evaluating the model, which is especially helpful if EarlyStopping was used
    # and the final model didn't reach the specified last epoch.
    if(finishTraining or not pathlib.Path(modelfile).exists()):   
        history = regressor.fit(
            x=x_train,
            y=y_train,
            validation_data=(
                x_valid,
                y_valid
            ),
            sample_weight=sample_weight,
            callbacks=callbacks
        )        
        
    saveModel = True
    if(initial_epoch == epochs or not finishTraining): saveModel = False
    if(saveModel):
        print('  Saving model to {}.'.format(modelfile))
        regressor.model.save(modelfile)
    
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

    return regressor, history

# Get resu
def GetPredictions(regressor, model_input, indices=None, truth=None, reco=None, scaler=None, mapping=None, filename=None):
    
    result = regressor.predict(model_input)
    if(scaler is not None): result = scaler.inverse_transform(result)
    if(mapping is not None): result = mapping.Inverse(result)
    
    # save the predictions to a file
    if(filename is not None):
        f = h5.File(filename,'w')
        
        # save the training/testing/validation indices if given (these might be useful to interpret results)
        if(indices is not None):
            for key in indices.keys():
                d = f.create_dataset(key,data=indices[key],chunks=True,compression='gzip',compression_opts=7)
                
        # save the truth values (to compare to predictions). This will definitely be useful (and required for some of our scripts to eval results)
        if(truth is not None):
            d = f.create_dataset('truth',data=truth,chunks=True,compression='gzip',compression_opts=7)
            
        # save the reco values (to compare/use with predictions). This will be useful as a baseline.
        if(reco is not None):
            d = f.create_dataset('reco',data=reco,chunks=True,compression='gzip',compression_opts=7)
            
        # save the network outputs
        d = f.create_dataset('output',data=result,chunks=True,compression='gzip',compression_opts=7)
        f.close()
    return result