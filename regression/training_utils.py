import sys, os, pickle
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model
path_prefix = os.getcwd() + '/../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import qol_util as qu
from util import ml_util as mu

# Given one set of input, indices and a model filename,
# train a single model.
# Suffix is used in practice to label as "charged" or "neutral" regression,
# whereas model_key is used to distinguish type/architecture.
def TrainNetwork(regressor, x_train, y_train, x_valid, y_valid, model_key, suffix, modelpath, sample_weight, saveModel=True, modelfile=None):
    model_dir = '/'.join([modelpath, model_key])
    try: os.makedirs(model_dir)
    except: pass
    
    if(modelfile != None):
        model_filename = modelfile
        history_filename = model_filename.replace('.h5','.history')
    else:
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
def LoadNetwork(regressor, model_key, suffix, modelpath, modelfile=None):
    model_dir = '/'.join([modelpath, model_key])
    
    if(modelfile != None):
        model_filename = modelfile
        history_filename = modelfile.replace('.h5','.history')
    
    else:
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
    
    if('modelfile' in kwargs.keys()): modelfile = kwargs['modelfile']
    else: modelfile = None
    
    if(loadModel):
        history = LoadNetwork(regressor, model_key, suffix, modelpath, modelfile=modelfile)
        
    else:
        x_train   = kwargs['x_train']
        y_train = kwargs['y_train']
        x_valid   = kwargs['x_valid']
        y_valid = kwargs['y_valid']
        sample_weight = kwargs['sample_weight']
        saveModel = kwargs['saveModel']
        history = TrainNetwork(regressor, x_train, y_train, x_valid, y_valid, model_key, suffix, modelpath, sample_weight, saveModel, modelfile=modelfile)
    return history


# --- Data Prep Functions below ---

# Prepare the calo images for input to training.
def LoadCaloImages(dtree,indices=-1,layers=['EMB1','EMB2','EMB3','TileBar0','TileBar1','TileBar2']):
    l = len(layers) * len(dtree.keys())
    i = 0
    pfx = 'Loading calo images:      '
    sfx = 'Complete'
    bl = 50
    qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)

    calo_images = {}
    for key in dtree.keys():
        calo_images[key] = {}
    
        for layer in layers:
            if(indices != -1): calo_images[key][layer] = mu.setupCells(dtree[key],layer, indices = indices[key])
            else: calo_images[key][layer] = mu.setupCells(dtree[key],layer)
            i += 1
            qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)
    return calo_images

# Prepare the combined input -- this is used for our "all" DNN model.
def CombinedInput(dframe, dtree, indices=-1, input_keys = ['s_logE','s_eta'], layers=['EMB1','EMB2','EMB3','TileBar0','TileBar1','TileBar2'], calo_images=None):
    if(calo_images is None): calo_images = LoadCaloImages(dtree,indices,layers)
    # Concatenate images, and prepare our combined input.
    All_input = {}
    keys = list(calo_images.keys())
    l = 3 * len(keys)
    i = 0
    pfx = 'Preparing combined input: '
    sfx = 'Complete'
    bl = 50
    qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)

    for key in keys:
        combined_images = np.concatenate(tuple([calo_images[key][layer] for layer in layers]), axis=1)
        i = i + 1
        qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)

        s_combined,scaler_combined = mu.standardCells(combined_images, layers)
        i = i + 1
        qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)
        
        All_input[key] = np.column_stack([dframe[key][x][indices[key]] for x in input_keys] + [s_combined])
        i = i + 1
        qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)
    return All_input


def ResnetInput(dframe, dtree, indices, layers, cell_shapes, input_keys=['s_logE','s_eta'], calo_images=None):
    if(calo_images is None): calo_images=LoadCaloImages(dtree,indices,layers)
    rn_input = calo_images.copy()

    # Unflatten images. Note that the key names match those defined within resnet model in models.py, which are currently hard-coded.
    for key,imageset in rn_input.items():
        rn_input[key] = {'input' + str(i):imageset[layer].reshape(tuple([-1] + list(cell_shapes[layer]))) for i,layer in enumerate(layers)}
        rn_input[key]['energy'] = dframe[key][input_keys[0]].to_numpy()
        rn_input[key]['eta'   ] = dframe[key][input_keys[1]].to_numpy()
    return rn_input

def DepthInput(dframe, dtree, indices, layers, input_keys=['s_logE','s_eta'], calo_images=None):
    if(calo_images is None): calo_images=LoadCaloImages(dtree,indices,layers)
    calo_depths = {key:np.zeros((len(indices[key]),6)) for key in calo_images.keys()}
    for key in calo_depths.keys():
        for i,layer in enumerate(layers):
            calo_depths[key][:,i] = np.sum(calo_images[key][layer],axis=1)
        norms = np.sum(calo_depths[key],axis=1)
        norms[norms==0.] = 1.
        calo_depths[key] = calo_depths[key] / norms[:,None]
        
    All_input = {}
    for key in calo_images.keys():
        All_input[key] = {}
        All_input[key]['energy'] = dframe[key][input_keys[0]].to_numpy()
        All_input[key]['eta'   ] = dframe[key][input_keys[1]].to_numpy()
        All_input[key]['depth' ] = calo_depths[key]
    return All_input
    
# Splitting things into training and validation -- if we use a dictionary structure it's a bit more involved than for CombinedInput.
def DictionarySplit(rn_input, training_indices, validation_indices):
    # Now explicitly split things up into training and validation data.
    rn_train = {
        key:{
            input_key:val[training_indices[key]] for input_key,val in dset.items()
        }
        for key,dset in rn_input.items()
    }
    rn_valid = {
        key:{
            input_key:val[validation_indices[key]] for input_key,val in dset.items()
        }
        for key,dset in rn_input.items()
    }
    return {'train':rn_train, 'valid':rn_valid}