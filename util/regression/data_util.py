import numpy as np
from util import qol_util as qu
from util import ml_util as mu

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
        combined_images = mu.standardCells(pcells[key],layer=layers)
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

# def LorentzInput(pdata, pcells, layers=None, branch_map={}, use_layer_names=False):
    
#     rn_input = ResnetInput(pdata,pcells,layers,branch_map,use_layer_names=True)
#     keys = rn_input.keys()
#     layers = list(mu.cell_meta.keys())
#     layer_keys = ['input_{}'.format(x) for x in range(len(layers))]
    
#     for key in keys: # e.g. [pp, p0]

#         for layer in layers:
#             print('TODO')
#     return