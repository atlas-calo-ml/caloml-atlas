import numpy as np
from util import qol_util as qu
from util import ml_util as mu
from numba import jit

# --- Data Prep Functions below ---

# Basic data preparation.
def DataPrep(pdata, trainfrac=0.7, filename=''):
    '''
    Inputs:
        - pdata:  A dictionary of DataFrames, whose keys correspond with dataset species (e.g. charged pion, neutral pion).
    '''
    # Create train/validation/test subsets containing 70%/10%/20%
    # of events from each type of pion event.
    # The resulting indices are stored within the DataFrames! (To use these for the pcells arrays, you must consult the DataFrame).
    for p_index, plabel in enumerate(pdata.keys()):
        mu.splitFrameTVT(pdata[plabel],trainfrac=trainfrac,key=plabel,filename='{}_indices.h5'.format(filename))
        pdata[plabel]['label'] = p_index
    return pdata

def CombinedInput(pdata, pcells, branches=[], layers=None):
    if(layers is None): layers = list(mu.cell_meta.keys())
    
    l = 2 * len(pdata.keys())
    i = 0
    pfx = 'Preparing combined input: '
    sfx = 'Complete'
    bl = 50 
    qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)

    All_input = {}
    for key,frame in pdata.items():
        # Combine images, so the array will have shape (nclusters, sum(n_pixels)).
        # Also apply a StandardScaler to pixels, across all layers.
        combined_images = mu.standardCells_new(pcells[key],layer=layers)
        i += 1
        qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)

        All_input[key] = np.column_stack([frame[x] for x in branches] + [combined_images])
        i += 1
        qu.printProgressBarColor(i, l, prefix=pfx, suffix=sfx, length=bl)
    return All_input

def ResnetInput(pdata, pcells, layers=None, branch_map={}, use_layer_names=False):
    if(layers is None): layers = list(mu.cell_meta.keys())
    
    # Unflatten images. Note the keys used here -- any network that uses this input must internally use the same keys to access the data.
    rn_input = {}
    
    for key,frame in pdata.items():
        
        if(use_layer_names):
            rn_input[key] = {
                layer:pcells[key][layer].reshape([-1,mu.cell_meta[layer]['len_eta'],mu.cell_meta[layer]['len_phi']])
                for i,layer in enumerate(layers)
            }
        else:
            rn_input[key] = {
                'input_{}'.format(i):pcells[key][layer].reshape([-1,mu.cell_meta[layer]['len_eta'],mu.cell_meta[layer]['len_phi']])
                for i,layer in enumerate(layers)
            }
            
        for branch,varname in branch_map.items(): rn_input[key][varname] = pdata[key][branch].to_numpy()
    
    # TODO: consider re-introducing some optional energy scaling of images.
    return rn_input

def DepthInput(pdata, pcells, layers=None, branch_map={}):
    if(layers is None): layers = list(mu.cell_meta.keys())
    nlayers = len(layers)
    calo_depths = {}
    for key,images in pcells.items():
        calo_depths[key] = np.array([np.sum(images[layer],axis=1) for layer in layers]).T # transpose to go from (nlayer, ncluster) to (ncluster,nlayer)
        norms = np.sum(calo_depths[key],axis=1)
        norms[norms==0.] = 1.
        calo_depths[key] = calo_depths[key] / norms[:,None]
    
    All_input = {}
    for key,frame in pdata.items():
        All_input[key] = {}
        All_input[key]['depth'] = calo_depths[key]
        for branch,varname in branch_map.items(): All_input[key][varname] = pdata[key][branch].to_numpy()
    return All_input
    
# Splitting things into training, validation and testing, when our data is in a dictionary format.
def DictionarySplit(rn_input, pdata, include_no_split=False):
    # Now explicitly split things up into training and validation data.
    
    splits = ['train','val','test']
    
    results = {
        split:{
            key:{
                input_key:val[pdata[key][split]] for input_key,val in dset.items()
            }
            for key,dset in rn_input.items()
        }
        for split in splits
    }
    
    # Optionally include the "unsplit" data, in case we still want this.
    if(include_no_split):
        results['all'] = {
            key:{
                input_key:val for input_key,val in dset.items()
            }
            for key,dset in rn_input.items()
        }
        
    return results
    
def LorentzInput(pdata, pcells, layers=None, branch_map={}, n_vecs = (10,10,8,4,4,2), use_layer_names=True, form='cartesian'):
    
    allowed_forms = ['cartesian','cylindrical']
    if(form not in allowed_forms):
        print('Error: form not in ',allowed_forms)
        assert(False)
    
    rn_input = ResnetInput(pdata,pcells,layers,branch_map,use_layer_names=True)
    keys = rn_input.keys()
    layers = list(mu.cell_meta.keys())
    if(use_layer_names): layer_keys = layers
    else: layer_keys = ['input_{}'.format(i) for i in range(len(layers))]
        
    lorentz_input = {}
    
    l = 0
    counter = 0
    for key in keys:
        for layer in layers:
            l += rn_input[key][layer].shape[0]
            
    pfx = 'Preparing Lorentz input: '
    sfx = 'Complete'
    bl = 50     
    qu.printProgressBarColor(counter, l, prefix=pfx, suffix=sfx, length=bl)
    
    for key in keys: # e.g. [pp, p0]
        lorentz_input[key] = {}
        eta_offsets = pdata[key]['clusterEta'].to_numpy()
        
        for varname in branch_map.values():
            lorentz_input[key][varname] = rn_input[key][varname]        
            
        for i,layer in enumerate(layers):
                        
            n_images = rn_input[key][layer].shape[0]
            vecs = np.zeros((n_images,n_vecs[i],4)) # pt,eta,phi,m=0
            
            # Get image dimensions.
            n_eta, n_phi = mu.cell_meta[layer]['len_eta'], mu.cell_meta[layer]['len_phi']
            d_eta, d_phi = mu.cell_meta[layer]['cell_size_eta'], mu.cell_meta[layer]['cell_size_phi']
            n_vecs_local = int(np.minimum(n_vecs[i], n_eta * n_phi))

            # Determine the distance of each pixel from the image center, in eta and phi.
            eta = np.linspace(0, n_eta * d_eta, n_eta)
            phi = np.linspace(0, n_phi * d_phi, n_phi)
            eta -= (n_eta / 2) * d_eta
            phi -= (n_phi / 2) * d_phi 
            
            for j,image in enumerate(rn_input[key][layer]):                
                eta_full = eta + eta_offsets[j]
                pt = image / np.cosh(eta_full)[:,None]
                
                # Now get the indices of the leading vectors.
                ind_x, ind_y = np.unravel_index(np.argsort(-pt, axis=None), pt.shape)
                ind_x, ind_y = ind_x[:n_vecs_local], ind_y[:n_vecs_local]
                
                # Fill the info.
                pt_j  = pt[(ind_x, ind_y)]
                eta_j = eta_full[ind_x]
                phi_j = phi[ind_y]
                e_j   = image[(ind_x, ind_y)]
                
                if(form == 'cartesian'):
                    # Now we have to convert to Cartesian
                    px_j = pt_j * np.cos(phi_j)
                    py_j = pt_j * np.sin(phi_j)
                    pz_j = pt_j * np.sinh(eta_j)
                    
                    vecs[j,:n_vecs_local,0] = px_j
                    vecs[j,:n_vecs_local,1] = py_j
                    vecs[j,:n_vecs_local,2] = pz_j                
                    vecs[j,:n_vecs_local,2] = e_j              
                    
                else:
                    vecs[j,:n_vecs_local,0] = pt[(ind_x, ind_y)]
                    vecs[j,:n_vecs_local,1] = eta_full[ind_x]
                    vecs[j,:n_vecs_local,2] = phi[ind_y]
                    # leaving the last component as zero
                    
                counter += 1
                if(counter%1000 == 0 or counter == l): qu.printProgressBarColor(counter, l, prefix=pfx, suffix=sfx, length=bl)
            lorentz_input[key][layer_keys[i]] = vecs
            
    return lorentz_input