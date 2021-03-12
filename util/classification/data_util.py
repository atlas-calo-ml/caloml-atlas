import numpy as np
import pandas as pd
from keras.utils import np_utils
from util import ml_util as mu

# Basic data preparation.
def DataPrep(pdata, pcells, layers, trainfrac=0.7):
    training_dataset = ['pi0','piplus'] # TODO: These are just the keys from pdata, but this enforces the ordering?
    # create train/validation/test subsets containing 70%/10%/20%
    # of events from each type of pion event
    for p_index, plabel in enumerate(training_dataset):
        mu.splitFrameTVT(pdata[plabel],trainfrac=0.7)
        pdata[plabel]['label'] = p_index

    # merge pi0 and pi+ events
    pdata_merged = pd.concat([pdata[ptype] for ptype in training_dataset])
    pcells_merged = {
        layer : np.concatenate([pcells[ptype][layer]
                                for ptype in training_dataset])
        for layer in layers
    }
    plabels = np_utils.to_categorical(pdata_merged['label'],len(training_dataset)) # higher score -> more likely to be charged
    
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