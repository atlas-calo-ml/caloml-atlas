import uproot as ur
import awkward as ak
import numpy as np


def loadVectorBranchFlat(branchName, tree):
    return np.copy(ak.flatten(tree[branchName].array()).to_numpy())

#given a branchname, a tree from uproot, and a padLength...
#return a flattened numpy array that flattens away the event index and pads cels to padLength
#if there's no cell, add a 0 value
def loadArrayBranchFlat(branchName, tree, padLength, cleanMask = False, debug = False):
    branchInfo = tree[branchName].array()

    # we flatten the event index, to generate a list of clusters
    branchFlat = ak.flatten(branchInfo)
    
    if(4591870180066957722 in branchFlat):
        print('weird ID is in flat branch')

    # pad the cell axis to the specified length
    branchFlatPad = ak.pad_none(branchFlat, padLength, axis=1)

    if(4591870180066957722 in branchFlatPad):
        print('weird ID is in padded')

    branchFlatShallow = branchFlatPad.to_numpy()
    if (4591870180066957722 in branchFlatShallow):
        print('weird ID in Numpy shallow')

    # branchFlatNumpy = np.copy(branchFlatShallow)
    branchFlatNumpy = np.ma.MaskedArray.copy(branchFlatShallow)
    # # Do a deep copy to numpy so that the data is owned by numpy
    #     # branchFlatNumpy = np.copy(branchFlatPad.to_numpy())

    # if (4591870180066957722 in branchFlatNumpy):
        # print('weird ID in Numpy')
    # if debug:
    #     print('Debugging!')
    #     for iterCluster, cluster in enumerate(branchFlatNumpy):
    #         for iterCell, cell in enumerate(cluster):
    #             if cell != branchFlatShallow[iterCluster][iterCell]:
    #                 print('{} {} {} {}'.format(iterCluster, iterCell,
    #                       cell, branchFlatShallow[iterCluster][iterCell]))


    # #replace the padding 'None' with 0's
    # make this configurable
    if cleanMask:
        branchFlatNumpy.soften_mask()
        branchFlatNumpy[-1] = 0 
        branchFlatNumpy.mask = np.ma.nomask
        # branchFlatShallow[-1] = 0 

    branchFlatNumpyReal = np.copy(branchFlatNumpy)

    # help out the garbage collection by deleting this early
    del branchFlat, branchFlatPad

    # return branchFlatShallow
    return branchFlatNumpyReal

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
        branchDict[4308257264] = mask # another magic safetey number? CHECKME
        
        globalDict[key] = branchDict

    return globalDict


# given a list of Cell IDs and a target from the geometry tree specified in geoString
# (and given the globalDict containing the ID->info mappings)
# return a conversion of the cell IDs to whatever is requested
def convertIDToGeo(cellID, geoString, globalDict):
    # MAGIC https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    count = 0
    fullCount = 0
    # if(geoString == 'cell_geo_sampling'):
    #     for cluster in cellID:
    #         for cell in cluster:
    #             fullCount += 1
    #             if cell not in globalDict[geoString]:
    #                 # print('I am not in the dictionary! {}'.format(cell))
    #                 count += 1
    print('Out of {} entries in cellID, {} are not in the dictionary'.format(fullCount, count))
    return np.vectorize(globalDict[geoString].get)(np.nan_to_num(cellID))
