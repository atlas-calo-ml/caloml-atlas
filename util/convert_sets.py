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
# data_path = '/fast_scratch/atlas_images/v01-45/'

data_path_pipm = data_path + 'pipm/'
data_path_pi0  = data_path + 'pi0/'

pipm_list = glob(data_path_pipm+'*root')
pi0_list =  glob(data_path_pi0 + '*root')

with ur.open(data_path + 'cell_geo.root') as file:
    geo_dict = gu.loadGraphDictionary(file['CellGeo'])


#pipm_list = ['/fast_scratch/atlas_images/v01-45/pipm/user.angerami.24559744.OutputStream._000421.root']
# pi0_list = ['/Users/swiatlow/Data/caloml/graph_data/pi0/user.angerami.24559740.OutputStream._000116.root']

def convertFile(filename, label, debug):
    print('Working on {}'.format(filename))
    
    with ur.open(filename) as file:

        tree = file['EventTree']
        # geotree = file['CellGeo']

        # geo_dict = gu.loadGraphDictionary(geotree)

        print('Loading data')
        # should I remove things over 2000?

        ## First, load all information we want
        cell_id = gu.loadArrayBranchFlat('cluster_cell_ID', tree, 2000, cleanMask = True)
        cell_e = gu.loadArrayBranchFlat('cluster_cell_E', tree, 2000)

        # np.save('cell_id.npy', cell_id)

        print('convert eta')
        cell_eta = gu.convertIDToGeo(cell_id, 'cell_geo_eta', geo_dict)
        print('convert phi')
        cell_phi = gu.convertIDToGeo(cell_id, 'cell_geo_phi', geo_dict)
        print('convert samp')
        cell_samp = gu.convertIDToGeo(cell_id, 'cell_geo_sampling', geo_dict)
        print('done converting')

        clus_phi = gu.loadVectorBranchFlat('cluster_Phi', tree)
        clus_eta = gu.loadVectorBranchFlat('cluster_Eta', tree)

        clus_e = gu.loadVectorBranchFlat('cluster_E', tree)

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
                    axis=2)

        Y_label = to_categorical(np.ones(len(X)) * label)
        Y_target = np.log(clus_targetE)

        #Now we save. prepare output filename.
        outname = filename.replace('root', 'npz')
        np.savez(outname, X=X, Y_label=Y_label, Y_target=Y_target)
        print('Done! {}'.format(outname))

        #unclear to me why this isn't cleaned up automatically, but OK
        #could be that garbage collection is being run less frequently than I want
        del cell_e, cell_eta, cell_id, cell_phi, cell_samp, clus_phi, clus_e, clus_eta, clus_targetE


for pipm_file in pipm_list:
    convertFile(pipm_file, 1, debug = False)

# pi0_list = ['/Users/swiatlow/Data/caloml/graph_data/pi0/user.angerami.24559740.OutputStream._000116.root']

for pi0_file in pi0_list:
    convertFile(pi0_file, 0, debug = False)



