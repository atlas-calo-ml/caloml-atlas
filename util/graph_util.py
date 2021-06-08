import uproot as ur
import awkward as ak
import numpy as np


def loadVectorBranchFlat(branchName, tree):
    return np.copy(ak.flatten(tree[branchName].array()).to_numpy())

#given a branchname, a tree from uproot, and a padLength...
#return a flattened numpy array that flattens away the event index and pads cels to padLength
#if there's no cell, add a 0 value
def loadArrayBranchFlat(branchName, tree, padLength):
    branchInfo = tree[branchName].array()

    # we flatten the event index, to generate a list of clusters
    branchFlat = ak.flatten(branchInfo)

    # pad the cell axis to the specified length
    branchFlatPad = ak.pad_none(branchFlat, padLength, axis=1)

    # # Do a deep copy to numpy so that the data is owned by numpy
    branchFlatNumpy = np.copy(branchFlatPad.to_numpy())

    # #replace the padding 'None' with 0's
    # branchFlatNumpy[-1] = 0 # Actually better to turn this off and mask later, so norms are easier

    return branchFlatNumpy

# A quick implemention of Dilia's idea for converting the geoTree into a dict
def loadGraphDictionary(tree):
    # make a global dict. this will be keyed by strings for which info you want
    globalDict = {}

    #get the information
    arrays = tree.arrays()
    keys = tree.keys()
    for key in keys:
        #skip geoID-- that's our new key
        if key == 'cell_geo_ID': 
            continue
        branchDict = {}
        # loop over the entries of the GEOID array (maybe this should be hte outer array? eh.)
        # [0] is used here and below to remove a false index
        for iter, ID in enumerate(arrays['cell_geo_ID'][0]):
            #the key is the ID, the value is whatever we iter over
            branchDict[ID] = arrays[key][0][iter] 
       

        if key == 'cell_geo_sampling':
            mask = 0
        else:
            mask = None

        branchDict[0] = mask
        branchDict[4308257264] = mask # another magic safetey number? CHECKM
            
        
        globalDict[key] = branchDict

    return globalDict


# given a list of Cell IDs and a target from the geometry tree specified in geoString
# (and given the globalDict containing the ID->info mappings)
# return a conversion of the cell IDs to whatever is requested
def convertIDToGeo(cellID, geoString, globalDict):
    # MAGIC https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    return np.vectorize(globalDict[geoString].get)(np.nan_to_num(cellID))
