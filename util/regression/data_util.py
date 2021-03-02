import numpy as np
from util import qol_util as qu
from util import ml_util as mu

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
# TODO: In practice, doesn't dframe already have indices applied?
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

def ResnetInput(dframe, dtree, indices, layers, cell_shapes, input_keys=['s_logE','s_eta'], calo_images=None, rescaling=True):
    if(calo_images is None): calo_images=LoadCaloImages(dtree,indices,layers)
    rn_input = calo_images.copy()

    # Unflatten images. Note that the key names match those defined within resnet model in models.py, which are currently hard-coded.
    for key,imageset in rn_input.items():
        rn_input[key] = {'input{}'.format(i):imageset[layer].reshape(tuple([-1] + list(cell_shapes[layer]))) for i,layer in enumerate(layers)}
        rn_input[key]['energy'] = dframe[key][input_keys[0]][indices[key]].to_numpy()
        rn_input[key]['eta'   ] = dframe[key][input_keys[1]][indices[key]].to_numpy()
        
    # Rescale images by our (scaled) energy, if requested. Their integrals already give the reco energy,
    # but as our "energy" input will have some EnergyMapping + sklearn scaler applied, we might want to 
    # achieve the same scaling on the images too. Rather than passing EnergyMapping and the scaler,
    # we can just do some rescaling using the "energy" input itself. We just have to be careful to preserve
    # the proportions of integrals of the different layers, as not to distort depth information.
    if(rescaling):
        for key in rn_input.keys():
            n = rn_input[key]['energy'].shape[0]
            integrals = np.array([np.sum(rn_input[key]['input{}'.format(x)],axis=(1,2)) for x in range(len(layers))])
            integrals = np.sum(integrals,axis=0)
            integrals[integrals == 0.] = 1.
            scale_factors = rn_input[key]['energy'] / integrals # element-wise division
            for i in range(len(layers)):
                rn_input[key]['input{}'.format(i)] = np.expand_dims(scale_factors, [1,2]) * rn_input[key]['input{}'.format(i)]
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