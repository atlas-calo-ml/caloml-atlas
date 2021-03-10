import numpy as np
import pandas as pd
from keras.utils import np_utils
from util import ml_util as mu

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