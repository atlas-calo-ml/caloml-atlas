import pathlib
import numpy as np  
import uproot as ur
import awkward as ak
import pandas as pd
import h5py as h5
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import utils
import scipy.ndimage as ndi
from util import qol_util as qu

#define a dict for cell meta data
cell_meta = {
    'EMB1': {
        'cell_size_phi': 0.098,
        'cell_size_eta': 0.0031,
        'len_phi': 4,
        'len_eta': 128
    },
    'EMB2': {
        'cell_size_phi': 0.0245,
        'cell_size_eta': 0.025,
        'len_phi': 16,
        'len_eta': 16
    },
    'EMB3': {
        'cell_size_phi': 0.0245,
        'cell_size_eta': 0.05,
        'len_phi': 16,
        'len_eta': 8
    },
    'TileBar0': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.1,
        'len_phi': 4,
        'len_eta': 4
    },
    'TileBar1': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.1,
        'len_phi': 4,
        'len_eta': 4
    },
    'TileBar2': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.2,
        'len_phi': 4,
        'len_eta': 2
    },
}

def createTrainingDatasets(categories, data, cells):
    # create train/validation/test subsets containing 70%/10%/20%
    # of events from each type of pion event
    for p_index, plabel in enumerate(categories):
        splitFrameTVT(data[plabel], trainfrac=0.7)
        data[plabel]['label'] = p_index

    # merge pi0 and pi+ events
    data_merged = pd.concat([data[ptype] for ptype in categories])
    cells_merged = {
        layer: np.concatenate([cells[ptype][layer] for ptype in categories])
        for layer in cell_meta
    }
    labels = utils.to_categorical(data_merged['label'], len(categories))

    return data_merged, cells_merged, labels

def reshapeSeparateCNN(cells):
    reshaped = {
        layer: cells[layer].reshape(cells[layer].shape[0], 1, cell_meta[layer]['len_eta'], cell_meta[layer]['len_phi'])
        for layer in cell_meta
    }

    return reshaped

def setupPionData(root_file_dict,branches=[], layers=[], cluster_tree='ClusterTree', 
                  balance_data=True, n_max=-1, 
                  match_distribution='', match_binning=(),
                  verbose=False, load=False, save=False, filename=''):

    indices = {}
    pdata = {}
    pcells = {}
    keys = list(root_file_dict.keys())
    rng = np.random.default_rng()

    pdata_filename = filename + '_frame.h5'
    pcell_filename = filename + '_images.h5'
    
    if(load and pathlib.Path(pdata_filename).exists() and pathlib.Path(pcell_filename).exists()):
        
        # Load the DataFrame and images from disk.
        pdata = {
            key: pd.read_hdf(pdata_filename,key=key)
            for key in keys
        }
        
        hf = h5.File(pcell_filename,'r')
        for key in keys:
            pcells[key] = {}
            for layer in layers:
#                 pcells[key][layer] = hf.get('{}_{}'.format(key,layer)).value # hf['{}:{}'.format(key,layer)]  
                pcells[key][layer] = hf['{}:{}'.format(key,layer)][:]
        hf.close()
        
    else:
        if(verbose): print('Preparing pandas DataFrame.')
        arrays = {
            key: ur.lazy(':'.join((rfile_match, cluster_tree)), branch_filter=lambda x: x.name in branches)
            for key,rfile_match in root_file_dict.items()        
        }

        # Create indices for selected clusters.
        for key in keys: indices[key] = np.full(len(arrays[key]),True,dtype=bool)
            
        # Optionally filter out clusters so that our data series match in their distribution of a user-supplied variable.
        if(match_distribution != ''):
            if(match_distribution in branches and len(match_binning) == 3):
                if(verbose): print('Matching data series on distribution: {}.'.format(match_distribution))
                        
                binning = np.linspace(match_binning[1],match_binning[2],match_binning[0]+1)
                n_bins = len(binning) - 1
                distributions = {
                    key: np.histogram(arrays[key][match_distribution].to_numpy(), bins=binning)[0] # only keep bin counts
                    for key in keys
                }
                
                # Now determine how many clusters we keep in each bin, for each key.
                n_keep = np.zeros(n_bins,dtype=np.dtype('i8'))

                for i in range(n_bins):
                    n_keep[i] = int(np.min([x[i] for x in distributions.values()]))
                    
                # Now we need to throw out some clusters -- in other words, only keep some.
                # We will randomly choose which ones we keep, for each match_distribution bin,
                # for each data series (key).
                for key in keys:
                    sorted_indices = np.argsort(arrays[key][match_distribution])
                    keep_indices = []
                    bin_idx_edges = np.insert(np.cumsum(distributions[key]),0,0)
                    for i in range(n_bins):
                        index_block = sorted_indices[bin_idx_edges[i]:bin_idx_edges[i+1]] # all indices corresponding to the i'th bin of match_distribution, for this key
                        keep_indices.append(rng.choice(index_block, n_keep[i], replace=False))
                    n_before = len(indices[key])
                    indices[key] = np.hstack(keep_indices)
                    n_after = len(indices[key])
                    if(verbose): print('\t{}, number of events: {} -> {}'.format(key, n_before, n_after))
                    
                    # check distribution
                    distributions[key] = np.histogram(arrays[key][match_distribution][indices[key]].to_numpy(), bins=binning)[0] # only keep bin counts
                    
            else: print('Warning: Requested matching of distribution \"{}\" but this variable is not among the branches you selected from the data. Skipping this step.'.format(match_distribution))            
            
        # Optionally balance data so we have equal amounts of each category.
        if(balance_data):
            n_max_tmp = np.min([len(x) for x in indices.values()])
            if(n_max > 0): n_max = np.minimum(n_max_tmp, n_max)
            else: n_max = n_max_tmp
            
            indices = {key:rng.choice(val, n_max, replace=False) for key,val in indices.items()}
            for key in indices.keys():
                msk = np.zeros(len(arrays[key]),dtype=np.bool)
                msk[indices[key]] = True
                indices[key] = msk
        arrays = {
            key:arrays[key][indices[key]]
            for key in keys
        }
        
        # Making dataframes.
        pdata = {
            key: ak.to_pandas(arrays[key][branches])
            for key in keys
        }
    
        arrays = {
            key: ur.lazy(':'.join((rfile_match, cluster_tree)), branch_filter=lambda x: x.name in layers)[indices[key]]
            for key,rfile_match in root_file_dict.items()        
        }   
        
        nentries = len(keys) * len(layers)
        i = 0
        if(verbose): qu.printProgressBarColor (i, nentries, prefix='Preparing calorimeter images.', suffix='% Complete', length=40)

        pcells = {}
        for key in keys:
            pcells[key] = {}
            for layer in layers:
                pcells[key][layer] = setupCells(arrays[key],layer)
                i+=1
                if(verbose): qu.printProgressBarColor (i, nentries, prefix='Preparing calorimeter images.', suffix='% Complete', length=40)
        
        if(filename != '' and save):
            if(verbose): print('Saving DataFrames to {}.'.format(pdata_filename))
            for key,frame in pdata.items():
                frame.to_hdf(pdata_filename, key=key, mode='a',complevel=6)   
                
            if(verbose): print('Saving calorimeter images to {}.'.format(pcell_filename))
                
            hf = h5.File(pcell_filename, 'w')
            for key in pcells.keys():
                for layer in layers:
                    dset = hf.create_dataset('{}:{}'.format(key,layer), data=pcells[key][layer], chunks=True, compression='gzip', compression_opts=7)
            hf.close()
                        
#     # We also want a container for the calorimeter images themselves.
#     if(verbose): print('Preparing cluster images.')
#     pcells = {
#         key: {
#             layer: setupCells(arrays[key],layer)
#             for layer in layers
#         }
#         for key in arrays.keys()
#     }
    
    return pdata, pcells

def splitFrameTVT(frame, trainlabel='train', trainfrac = 0.8, testlabel='test', testfrac = 0.2, vallabel='val'):

    valfrac = 1.0 - trainfrac - testfrac
    
    train_split = ShuffleSplit(n_splits=1, test_size=testfrac + valfrac, random_state=0)
    # advance the generator once with the next function
    train_index, testval_index = next(train_split.split(frame))  

    if valfrac > 0:
        testval_split = ShuffleSplit(
            n_splits=1, test_size=valfrac / (valfrac+testfrac), random_state=0)
        test_index, val_index = next(testval_split.split(testval_index)) 
    else:
        test_index = testval_index
        val_index = []
        
    frame[trainlabel] = frame.index.isin(train_index)
    frame[testlabel]  = frame.index.isin(test_index)
    frame[vallabel]   = frame.index.isin(val_index)

def setupCells(arrays, layer, nrows = -1, indices = [], flatten=True):
    if(type(arrays) != list): arrays = [arrays]
    array = np.row_stack([arr[layer].to_numpy() for arr in arrays])
    if nrows > 0:
        array = array[:nrows]
    elif len(indices) > 0:
        array = array[indices]
    num_pixels = cell_meta[layer]['len_phi'] * cell_meta[layer]['len_eta']
    if flatten:
        array = array.reshape(len(array), num_pixels)
    
    return array

def standardCells(array, layer, nrows = -1):
    if nrows > 0:
        working_array = array[:nrows]
    else:
        working_array = array

    scaler = StandardScaler()
    if type(layer) == str:
        num_pixels = cell_meta[layer]['len_phi'] * cell_meta[layer]['len_eta']
    elif type(layer) == list:
        num_pixels = 0
        for l in layer:
            num_pixels += cell_meta[l]['len_phi'] * cell_meta[l]['len_eta']
    else:
        print('you should not be here')

    num_clusters = len(working_array)

    flat_array = np.array(working_array.reshape(num_clusters * num_pixels, 1))


    scaled = scaler.fit_transform(flat_array)

    reshaped = scaled.reshape(num_clusters, num_pixels)
    return reshaped, scaler

def standardCellsGeneral(array, nrows = -1):
    if nrows > 0:
        working_array = array[:nrows]
    else:
        working_array = array

    scaler = StandardScaler()

    shape = working_array.shape

    total = 1
    for val in shape:
        total*=val

    flat_array = np.array(working_array.reshape(total, 1))

    scaled = scaler.fit_transform(flat_array)

    reshaped = scaled.reshape(shape)
    return reshaped, scaler


#rescale our images to a common size
#data should be a dictionary of numpy arrays
#numpy arrays are indexed in cluster, eta, phi
#target should be a tuple of the targeted dimensions
#if layers isn't provided, loop over all the layers in the dict
#otherwise we just go over the ones provided
def rescaleImages(data, target, layers = []):
    if len(layers) == 0:
        layers = data.keys()
    out = {}
    for layer in layers:
        out[layer] = ndi.zoom(data[layer], (1, target[0] / data[layer].shape[1], target[1] / data[layer].shape[2]))

    return out

#just a quick thing to stack things along axis 1, channels = first standard for CNN
def setupChannelImages(data,last=False):
    axis = 1
    if last:
        axis = 3
    return np.stack([data[layer] for layer in data], axis=axis)


def rebinImages(data, target, layers = []):
    '''
    Rebin images up or down to target size
  
    :param data: A dictionary of numpy arrays, numpy arrays are indexed in cluster, eta, phi
    :param target: A tuple of the targeted dimensions
    :param layers: A list of the layers to be rebinned, otherwise loop over all layers
    :out: Dictionary of arrays whose layers have been rebinned to the target size
    '''
    if len(layers) == 0:
        layers = data.keys()
    out = {}
    for layer in layers:
        shape = data[layer].shape
        # First rebin eta up or down as needed
        if target[0] <= shape[1]:
            out[layer] = [rebinDown(cluster, target[0], shape[2]) for cluster in data[layer]]
        elif target[0] > shape[1]:
            out[layer] = [rebinUp(cluster, target[0], shape[2]) for cluster in data[layer]]  
            
        # Next rebin phi up or down as needed
        if target[1] <= shape[2]:
            out[layer] = [rebinDown(cluster, target[0], target[1]) for cluster in out[layer]]
        elif target[1] > shape[2]:
            out[layer] = [rebinUp(cluster, target[0], target[1]) for cluster in out[layer]]

    return out

def rebinDown(a, targetEta, targetPhi):
    '''
    Decrease the size of a to the dimensions given by targetEta and targetPhi. Target dimensions must be factors of dimensions of a. Rebinning is done by summing sets of n cells where n is factor in each dimension.
    
    :param a: Array to be rebinned
    :param targetEta: End size of eta dimension
    :param targetPhi: End size of phi dimension
    :out: Array rebinned to target size
    '''
    # Get shape of existing array
    shape = a.shape
    
    # Calcuate factors by which we're reducing each dimension and check that they're integers
    etaFactor = shape[0] / targetEta
    if etaFactor != int(etaFactor):
        raise ValueError('Target eta dimension must be integer multiple of current dimension')
    phiFactor = shape[1] / targetPhi
    if phiFactor != int(phiFactor):
        raise ValueError('Target phi dimension must be integer multiple of current dimension')
        
    # Perform the reshaping and summing to get to target shape
    a = a.reshape(targetEta, int(etaFactor), targetPhi, int(phiFactor),).sum(1).sum(2)
    
    return a

def rebinUp(a, targetEta, targetPhi):
    '''
    Increase the size of a to the dimensions given by targetEta and targetPhi. Target dimensions must be integer multiples of dimensions of a. The value of a cell is divided equally amongst the new cells taking its place.
    
    :param a: Array to be rebinned
    :param targetEta: End size of eta dimension
    :param targetPhi: End size of phi dimension
    :out: Array rebinned to target size
    '''
    # Get shape of existing array
    shape = a.shape
    
    # Calculate factors by which we're expanding each dimension and check that they're integers
    etaFactor = targetEta / shape[0]
    if etaFactor != int(etaFactor):
        raise ValueError('Target eta dimension must be integer multiple of current dimension')
    phiFactor = targetPhi / shape[1]
    if phiFactor != int(phiFactor):
        raise ValueError('Target phi dimension must be integer multiple of current dimension')
        
    # Apply upscaling
    a = upscaleEta(a, int(etaFactor))
    a = upscalePhi(a, int(phiFactor))
    
    return a

def upscalePhi(array, scale):
    '''
    Upscale an array along the phi axis (index 1) by calling upscaleList on row
    
    :param array: 2D array to be upscaled
    :param scale: Positive integer, the factor by which to increase the size of array in the phi direction
    :out: Upscaled array
    '''
    out_array = np.array([upscaleList(row, scale) for row in array])
    return out_array
    
def upscaleEta(array, scale):
    '''
    Upscale an array along the eta axis (index 0) by flipping eta and phi, calling upscalePhi on each row, and flipping back
    
    :param array: 2D array to be upscaled
    :param scale: Positive integer, the factor by which to increase the size of array in the eta direction
    :out: Upscaled array
    '''
    transpose_array = array.T
    out_array = upscalePhi(transpose_array, scale)
    out_array = out_array.T
    return out_array
    
def upscaleList(val_list, scale):
    '''
    Expand val_list by the scale multiplier. Each element of val_list is replaced by scale copies of that element divided by scale.
    E.g. upscaleList([3, 3], 3) = [1, 1, 1, 1, 1, 1]
    
    :param val_list: List to be upscaled
    :param scale: Positive integer, the factor by which to increase the size of val_list
    :out: Upscaled list
    '''
    if scale >= 1:
        if scale != int(scale):
            raise ValueError('Scale must be an integer')
        out_list = [val / scale for val in val_list for _ in range(scale)]
    else:
         raise ValueError('Scale must be greater than or equal to one')
    return out_list

class cell_info:
    '''
    Convenience accessor for retrieving cell information via the 'cluster_cell_ID' hash.
    The constructor takes a path to a root file containing the 'CellGeo' tree as its only argument.
    Given the 'cluster_cell_ID' hash for a cell, retrieve its information by indexing a cell_info object with that hash; for example:
      ci = cell_info('inputfile.root')
      ci[1149470720] # hash for a cell in TileBar0 (cell_geo_sampling=12)
    Alternatively, you can use the member functions 'get_cell_info' or 'get_cell_info_vector' directly by passing them the hash as their only argument.
    '''
    meta_tree = 'CellGeo'
    id_branch = 'cell_geo_ID'
    
    def __init__(self, metafile):
        with ur.open(metafile) as ifile:
            self.meta_keys = ifile[self.meta_tree].keys()
            self.celldata = ifile[self.meta_tree].arrays(
                self.meta_keys)
            
        self.id_map = {}
        for i, cell_id in enumerate(self.celldata[self.id_branch][0]):
            self.id_map[cell_id] = i

    def get_cell_info(self, cell_id):
        return {
            k : self.celldata[k][0][self.id_map[cell_id]]
            for k in self.meta_keys
        }
    
    def get_cell_info_vector(self, cell_id):
        res = []
        for k in self.meta_keys:
            if(k == self.id_branch):
                continue
            res.append(self.celldata[k][0][self.id_map[cell_id]])
        return res
    
    def __getitem__(self, key):
        return self.get_cell_info(key)

def create_cell_images(input_file, sampling_layers, c_info=None,
                       eta_range=0.4, phi_range=0.4, print_frequency=100, 
                       entries=-1, prefix = ''):

    '''Generates images from a 'graph' format input file.
    The output is a dictionary with the following structure:
      images[layer][event_index][eta_index][phi_index]
    The arguments are as follows:
      input_file: path to the desired input file
      sampling_layers: a dict which specifies which layers should
                       have images generated for them; this dict
                       should have entries of the form
                         (int)cell_geo_sampling : 'LayerName'
      c_info: either a path to a root file which contains the
              'CellGeo' tree, or a cell_info object; defaults
              to using input_file to create a cell_info object
              if not provided
      eta/phi_range: full width of the 'window' around cluster
                     centres to render images in; cells outside
                     this window will be ignored
      print_frequency: progress printout will be displayed every
                       integer multiple of this parameter
    '''
    
    if(c_info==None):
        ci = cell_info(input_file)
    elif(isinstance(c_info,str)):
        ci = cell_info(c_info)
    elif(isinstance(c_info,cell_info)):
        ci = c_info
    else:
        raise ValueError('Invalid argument for c_info: must be cell_info object or path to a root file with the CellGeo tree.')
    
    with ur.open(input_file) as ifile:
        if(entries < 0): entries = ifile['EventTree'].num_entries

        pdata = ifile['EventTree'].arrays(
            ['cluster_cell_ID', 'cluster_cell_E', 'cluster_E', 'cluster_Eta', 'cluster_Phi'])
    
    eta_min = -1*eta_range/2.0
    phi_min = -1*phi_range/2.0
    
    pcells = {
        layer : np.zeros((entries,meta['len_eta'],meta['len_phi']))
        for layer,meta in cell_meta.items()
    }
    qu.printProgressBarColor (0, entries, prefix=prefix, suffix='% Complete', length=50)
    
    for evt in range(entries):
        if((evt)%print_frequency==0 or evt==entries-1):
            qu.printProgressBarColor (evt+1, entries, prefix=prefix, suffix='% Complete', length=50)
        for clus in range(len(pdata['cluster_cell_ID'][evt])):
            for cell in range(len(pdata['cluster_cell_ID'][evt][clus])):
                
                c_info = ci[pdata['cluster_cell_ID'][evt][clus][cell]]
                
                if c_info['cell_geo_sampling'] in sampling_layers:
                    layer = sampling_layers[c_info['cell_geo_sampling']]
                    c_eta = pdata['cluster_Eta'][evt][clus]
                    c_phi = pdata['cluster_Phi'][evt][clus]

                    # calculate eta/phi bins using the formula
                    #   bin = floor( (x-x_min) * nbins / x_range )
                    eta_bin = int(
                        (c_info['cell_geo_eta']-c_eta-eta_min) *
                        cell_meta[layer]['len_eta'] / eta_range
                    )
                    phi_bin = int(
                        (c_info['cell_geo_phi']-c_phi-phi_min) *
                        cell_meta[layer]['len_phi'] / phi_range
                    )

                    # discard cells outside the eta/phi window
                    if(eta_bin<0 or
                       eta_bin>=cell_meta[layer]['len_eta'] or
                       phi_bin<0 or
                       phi_bin>=cell_meta[layer]['len_phi']):
                        continue

                    pcells[layer][evt][eta_bin][phi_bin] += pdata['cluster_cell_E'][evt][clus][cell] / pdata['cluster_E'][evt][clus]
                    # note: 'cluster_E' includes energies from cells with <5 MeV, which are not
                    # included in some datasets, so the energy fraction may be slightly off

    return pcells
