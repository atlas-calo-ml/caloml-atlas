import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from util import ml_util as mu

# Basic data preparation.
def DataPrep(pdata, pcells, layers, trainfrac=0.7, filename=''):
    
    # Create train/validation/test subsets containing 70%/10%/20%
    # of events from each type of pion event.
    # The resulting indices are stored within the DataFrames! (To use these for the pcells arrays, you must consult the DataFrame).
    for p_index, plabel in enumerate(pdata.keys()):
        mu.splitFrameTVT(pdata[plabel],trainfrac=trainfrac,key=plabel,filename='{}_indices.h5'.format(filename))
        pdata[plabel]['label'] = p_index

    # merge pi0 and pi+ events
    pdata_merged = pd.concat([pdata[ptype] for ptype in pdata.keys()])
    pcells_merged = {
        layer : np.concatenate([pcells[ptype][layer] for ptype in pdata.keys()])
        for layer in layers
    }
    plabels = to_categorical(pdata_merged['label'],len(pdata.keys())) # higher score -> more likely to be charged
    return pdata_merged, pcells_merged, plabels

def ReshapeImages(pcells_merged, cell_shapes, use_layer_names=False, keys=[]):
    if(keys == []):
        keys = ['input{}'.format(i) for i in range(len(pcells_merged.keys()))]
    assert(len(keys) == len(pcells_merged.keys()))
    
    if(use_layer_names): pcells_merged_unflattened = {key:pcells_merged[key].reshape(tuple([-1] + list(cell_shapes[key]))) for i,key in enumerate(pcells_merged.keys())}
    else: pcells_merged_unflattened = {keys[i]:pcells_merged[key].reshape(tuple([-1] + list(cell_shapes[key]))) for i,key in enumerate(pcells_merged.keys())}
    return pcells_merged_unflattened
        
# Split a dictionary of inputs into training, validation and testing samples. E.g. useful for ReshapeImages output
def DictionarySplit(pcells_merged_unflattened, train_indices, validation_indices, test_indices):
    train = {key:val[train_indices     ] for key,val in pcells_merged_unflattened.items()}
    valid = {key:val[validation_indices] for key,val in pcells_merged_unflattened.items()}
    test  = {key:val[test_indices      ] for key,val in pcells_merged_unflattened.items()}
    return {'train': train, 'valid':valid, 'test':test}