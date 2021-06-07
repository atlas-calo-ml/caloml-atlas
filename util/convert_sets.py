import uproot as ur
import awkward as ak
import numpy as np
from glob import glob

from tensorflow.keras.utils import to_categorical

import sys
sys.path.append('/Users/swiatlow/Code/ML4P/LCStudies')
sys.path.append('/home/mswiatlowski/start_tf/LCStudies')
from  util import graph_util as gu

data_path = '/Users/swiatlow/Data/caloml/graph_data/'
data_path = '/fast_scratch/atlas_images/v01-45/'

data_path_pipm = data_path + 'pipm/'
data_path_pi0  = data_path + 'pi0/'

pipm_list = glob(data_path_pipm+'*root')
pi0_list =  glob(data_path_pi0 + '*root')

pipm_list = ['/fast_scratch/atlas_images/v01-45/pipm_medium.root']
pi0_list = ['/fast_scratch/atlas_images/v01-45/pi0_medium.root']

def convertFile(filename, label):
    print('Working on {}'.format(filename))
    
    tree = ur.open(filename)['EventTree']
    geotree = ur.open(filename)['CellGeo']

    geo_dict = gu.loadGraphDictionary(geotree)

    print('Loading data')
    # should I remove things over 2000?

    ## First, load all information we want
    cell_id = gu.loadArrayBranchFlat('cluster_cell_ID', tree, 2000)
    cell_e  = gu.loadArrayBranchFlat('cluster_cell_E', tree, 2000)

    cell_eta  = gu.convertIDToGeo(cell_id, 'cell_geo_eta', geo_dict)
    cell_phi  = gu.convertIDToGeo(cell_id, 'cell_geo_phi', geo_dict)
    cell_samp = gu.convertIDToGeo(cell_id, 'cell_geo_sampling', geo_dict)

    clus_phi = gu.loadVectorBranchFlat('cluster_Phi', tree)
    clus_eta = gu.loadVectorBranchFlat('cluster_Eta', tree)

    clus_e   = gu.loadVectorBranchFlat('cluster_E', tree)

    clus_targetE = gu.loadVectorBranchFlat('cluster_ENG_CALIB_TOT', tree)

    ## Now, setup selections
    eta_mask = abs(clus_eta) < 0.7
    e_mask = clus_e > 0.5

    selection = eta_mask & e_mask

    ## Now, normalize
    print('Normalizing')    
    # normalize cell location relative to cluster center
    cell_eta = np.nan_to_num(cell_eta - clus_eta[:, None])
    cell_phi = np.nan_to_num(cell_phi - clus_phi[:, None])
    #normalize energy by taking log
    cell_e = np.nan_to_num(np.log(cell_e))
    #normalize sampling by 0.1
    cell_samp = cell_samp * 0.1

    print('Writing out')
    #prepare outputs
    X = np.stack((cell_e[selection],
                    cell_eta[selection],
                    cell_phi[selection]),
                    axis = 2)

    Y_label = to_categorical(np.ones(len(X)) * label)
    Y_target = np.log(clus_targetE[selection])

    #Now we save. prepare output filename.
    outname = filename.replace('root', 'npz')
    np.savez(outname, X=X, Y_label=Y_label, Y_target=Y_target)
    print('Done! {}'.format(outname))


for pipm_file in pipm_list:
    convertFile(pipm_file, 1)

for pi0_file in pi0_list:
    convertFile(pi0_file, 0)



