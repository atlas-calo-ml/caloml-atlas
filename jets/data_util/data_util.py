# Some data utilities for easy access.
import sys, os, pathlib
import numpy as np
import h5py as h5
path_prefix = os.getcwd() + '/../../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)

from util import ml_util as mu # Data preparation and wrangling for our neural networks.
from util import qol_util as qu # Quality-of-life stuff, like plot styles and progress bars.

from util.classification import data_util as cdu
from util.regression import data_util as rdu

classifier_modelnames = ['cnn_split_EMB']
regressor_modelnames = ['baseline_nn','simple_dnn','simple_cnn','split_emb_cnn','lorentz']

def ClassifierDataPrep(modelname, pdata, pcells, filename=None):
    
    if(modelname not in classifier_modelnames):
        raise ValueError('modelname not recognized.')
        
    if(filename is not None and pathlib.Path(filename).exists()):
        print('Loading {}.'.format(filename))
        f = h5.File(filename,'r')
        dkeys = list(f.keys())
        classifier_input = {dkey: f[dkey][:] for dkey in dkeys}
        return classifier_input
        
    if(modelname == 'cnn_split_EMB'):
        cell_shapes = {key: (val['len_eta'],val['len_phi']) for key,val in mu.cell_meta.items()}
        classifier_input = cdu.ReshapeImages(pcells, cell_shapes, use_layer_names=True)     
        
    # Save to file.
    if(filename is not None and not pathlib.Path(filename).exists()):
        f = h5.File(filename,'w')
        for key in classifier_input.keys():
            dset = f.create_dataset(key,data=classifier_input[key], compression='gzip',compression_opts=7)
        f.close()
        
    return classifier_input

def RegressorDataPrep(modelname, pdata, pcells, **kwargs):
    
    if(modelname not in regressor_modelnames):
        raise ValueError('modelname not recognized.')
        
    bin_edges = kwargs['reco_energy_bin_edges']
    s_var_prefixes = kwargs['scaled_variable_prefixes']
    reg_indxs = kwargs['regressor_indices']
    
    filename = None
    if('filename' in kwargs.keys()):
        filename = kwargs['filename']
        
    regressor_input = []
        
    if(filename is not None and pathlib.Path(filename).exists()):
        print('Loading {}.'.format(filename))
        
        f = h5.File(filename,'r')
        dkeys = list(f.keys())
        
        # Determine if we're dealing with a list of arrays, or a list of dictionary of arrays.
        
        # Case 1: list of arrays
        if(dkeys[0][0] == 'a'):
            dkeys.sort() # sort the keys
            for i,dkey in enumerate(dkeys):
                regressor_input.append(f[dkey][:])
            
        else:
            dkeys.sort() # sort the dkeys
            #print(dkeys)
            
            for i,dkey in enumerate(dkeys):
                num = int(dkey.split('_')[1])
                key = '_'.join(dkey.split('_')[2:])
                
                if(len(regressor_input) == num):
                    regressor_input.append({})
                regressor_input[-1][key] = f[dkey][:]
                
        return regressor_input

    dummy_key = 'jet'

    if(modelname == 'baseline_dnn'):
        
        for i in range(len(bin_edges)):
            prefix = s_var_prefixes[i]
            branches = ['{}_logE'.format(prefix), '{}_clusterEtaAbs'.format(prefix)]
            reg_input = rdu.CombinedInput(
                {dummy_key:pdata},
                {dummy_key:pcells},
                branches = branches
            )
            
            reg_input = reg_input[dummy_key]
            # We can immediately pare things down by removing clusters that won't be used by a particular regressor.
            reg_input = reg_input[reg_indxs[i]]
            regressor_input.append(reg_input)  
            
    elif(modelname == 'simple_dnn'):
        
        for i in range(len(bin_edges)):
            prefix = s_var_prefixes[i]
            reg_input = rdu.DepthInput(
                {dummy_key:pdata},
                {dummy_key:pcells},
                branch_map = {
                    '{}_logE'.format(prefix):'energy',
                    '{}_clusterEtaAbs'.format(prefix):'eta'
                }

            )
            reg_input = reg_input[dummy_key]
            # We can immediately pare things down by removing clusters that won't be used by a particular regressor.
            for key in reg_input.keys():
                reg_input[key] = reg_input[key][reg_indxs[i]]
            regressor_input.append(reg_input)

    elif(modelname == 'split_emb_cnn' or modelname == 'simple_cnn'):
                
        for i in range(len(bin_edges)):
            prefix = s_var_prefixes[i]
            reg_input = rdu.ResnetInput(
                {dummy_key:pdata},
                {dummy_key:pcells},
                branch_map = {
                    '{}_logE'.format(prefix):'energy',
                    '{}_clusterEtaAbs'.format(prefix):'eta'
                }
            )
            reg_input = reg_input[dummy_key]
            # We can immediately pare things down by removing clusters that won't be used by a particular regressor.
            for key in reg_input.keys():
                reg_input[key] = reg_input[key][reg_indxs[i]]
            regressor_input.append(reg_input)
            
    elif(modelname == 'lorentz'):
        
        for i in range(len(bin_edges)):
            prefix = s_var_prefixes[i]
            reg_input = rdu.LorentzInput(
                {dummy_key:pdata},
                {dummy_key:pcells},
                branch_map = {
                    '{}_logE'.format(prefix):'energy',
                    '{}_clusterEtaAbs'.format(prefix):'eta'
                },
                n_vecs = (8,8,6,4,2,2)  # TODO: don't hardcode this
            )
            reg_input = reg_input[dummy_key]
            # We can immediately pare things down by removing clusters that won't be used by a particular regressor.
            for key in reg_input.keys():
                reg_input[key] = reg_input[key][reg_indxs[i]]
            regressor_input.append(reg_input)
            
    # Optionally save the regressor input to a file for quick loading.
    if(filename is not None and not pathlib.Path(filename).exists()):
        N = len(regressor_input)
        f = h5.File(filename,'w')
        
        # We need to consider lists of dictionaries of arrays, and lists of arrays.
        t = type(regressor_input[0])
        
        # Case 1: list of arrays
        if(t == np.ndarray):
            for i in range(N):
                dkey = 'a_{}'.format(i)
                dset = f.create_dataset(dkey,data=regressor_input[i],compression='gzip',compression_opts=7)
        else:
            for i in range(N):
                key_prefix = 'k_{}'.format(i)
                for key,val in regressor_input[i].items():
                    dkey = '{}_{}'.format(key_prefix,key)
                    dset = f.create_dataset(dkey,data=regressor_input[i][key], compression='gzip',compression_opts=7)
        f.close()
             
    return regressor_input

