import os, pickle
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model

# Given one set of input, indices and a model filename,
# train a single model.
# Suffix is used in practice to label as "charged" or "neutral" regression,
# whereas model_key is used to distinguish type/architecture.
def TrainNetwork(regressor, x_train, y_train, x_valid, y_valid, model_key, suffix, modelpath, sample_weight, saveModel=True):
    model_dir = '/'.join([modelpath, model_key])
    try: os.makedirs(model_dir)
    except: pass
    
    model_filename   = '{}/{}{}.h5'.format(model_dir,model_key,suffix)
    history_filename = '{}/{}{}.history'.format(model_dir,model_key,suffix)
    history = regressor.fit(
        x=x_train,
        y=y_train,
        validation_data=(
            x_valid,
            y_valid
        ),
        sample_weight=sample_weight
    )
    history=history.history
    if(saveModel):
        print('  Saving model to {}.'.format(model_filename))
        regressor.model.save(model_filename)
        print('Saving history to {}.'.format(history_filename))
        with open(history_filename,'wb') as model_history_file:
            pickle.dump(history, model_history_file)
    return history

# Load a network -- note that you pass a regressor, which will be an empty wrapper.
# It will be filled with the actual saved network.
# Returns history.
def LoadNetwork(regressor, model_key, suffix, modelpath):
    model_dir = '/'.join([modelpath, model_key])
    model_filename   = '{}/{}{}.h5'.format(model_dir,model_key,suffix)
    history_filename = '{}/{}{}.history'.format(model_dir,model_key,suffix)
    history = 0
    print('     Loading model at {}.'.format(model_filename))
    regressor.model = load_model(model_filename)
    print('   Loading history at {}.'.format(history_filename))
    with open(history_filename,'rb') as model_history_file:
        history = pickle.load(model_history_file)
    return history
    
# Loads or trains a network
def PrepNetwork(regressor, model_key, suffix, modelpath, loadModel=False, **kwargs):    
    history = 0
    if(loadModel):
        history = LoadNetwork(regressor, model_key, suffix, modelpath)
        
    else:
        x_train   = kwargs['x_train']
        y_train = kwargs['y_train']
        x_valid   = kwargs['x_valid']
        y_valid = kwargs['y_valid']
        sample_weight = kwargs['sample_weight']
        saveModel = kwargs['saveModel']
        history = TrainNetwork(regressor, x_train, y_train, x_valid, y_valid, model_key, suffix, modelpath, sample_weight, saveModel)
    return history