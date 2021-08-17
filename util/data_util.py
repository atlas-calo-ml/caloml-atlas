# We implement a custom "data generator" that lets us stream data from ROOT files for TensorFlow.
# Here we use the tf.keras approach. If this turns out to be slow, there are some alternatives,
# e.g. see https://rubikscode.net/2019/12/09/creating-custom-tensorflow-dataset/ .

import sys, os, glob
import tensorflow as tf
import numpy as np
import ROOT as rt
import uproot as ur
from util import ml_util as mu

class ROOTImageArray():
    def __init__(self,
                 root_files,
                 tree_name,
                 image_branches = list(mu.cell_meta.keys()),
                 image_shapes = None
                ):
        
        self.chain = rt.TChain(tree_name)
        for rfile in root_files: self.chain.Add(rfile)
        self.branches = image_branches
        
        if(image_shapes is None): 
            self.image_shapes = [(mu.cell_meta[x]['len_eta'],mu.cell_meta[x]['len_phi']) for x in self.branches]
        else:  
            self.image_shapes = image_shapes
        assert(len(self.image_shapes) == len(self.branches))
        
        # Set up reading of the tree in the initialization.
        self.read_buffer = {
            br: np.zeros(self.image_shapes[i], dtype=np.dtype('f4')) # TODO: dtype is hard-coded
            for i,br in enumerate(self.branches)
        }        
        
        self.chain.SetBranchStatus('*',0)
        for br in self.branches: 
            self.chain.SetBranchStatus(br,1)
            self.chain.SetBranchAddress(br, self.read_buffer[br])

    def __getitem__(self, index):
        if(type(index) in [list,tuple,np.ndarray]):
            return self.get_item_many(index)
        status = self.chain.GetEntry(index)
        return self.read_buffer # TODO: should I use .copy()? Avoiding it might be faster.
    
    def get_item_many(self,indices):
        images = {}
        for i,idx in enumerate(indices):
            buff = self.__getitem__(idx)
            if(i == 0): 
                images = buff.copy()
                # Reshape to 3D (1st dim will be idx of image)
                for br in self.branches:
                    images[br] = images[br][None,...]                
            else:
                for br in self.branches:
                    images[br] = np.concatenate((images[br],buff[br][None,...]),axis=0)
        return images
    
    def get_types(self):
        return [x.dtype for x in self.read_buffer.values()]

class MLTreeV1DataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 root_files,
                 tree_name,
                 scalar_branches,
                 matrix_branches = list(mu.cell_meta.keys()),
                 target=None,
                 batch_size=200,
                 shuffle=True,
                 step_size=None):
        
        if(type(root_files) == list): 
            self.root_files = root_files
        else:
            self.root_files = glob.glob(root_files,recursive=True)
        
        if(step_size is None):
            self.step_size = '{} MB'.format(batch_size) # TODO: Is this reasonable?
        else: 
            self.step_size = step_size
        
        self.tree_name = tree_name
        self.scalar_branches = scalar_branches # We will create a lazy array for these, as it performs well.
        self.matrix_branches = matrix_branches # These will only be handled when fetching data! Not using lazy array (too slow).
        self.target = target
        if(self.target is not None): assert(self.target in self.scalar_branches)
        self.batch_size = batch_size
        self.shuffle = shuffle
                
        if(self.scalar_branches is None): filter_func = lambda x: x.name not in list(mu.cell_meta.keys())
        else: filter_func = lambda x: x.name in self.scalar_branches
        self.scalar_array = ur.lazy(files=[':'.join((x,self.tree_name)) for x in root_files],
                            filter_branch = filter_func,
                            step_size = self.step_size
                           )
        
        self.image_array = ROOTImageArray(root_files = self.root_files,
                                          tree_name = self.tree_name,
                                          image_branches = self.matrix_branches
                                         )
        
        self.indices = np.arange(len(self.scalar_array))
        self.on_epoch_end()
     
    # Gives the number of batches per epoch.
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    # Generate one batch of data.
    def __getitem__(self, index):
        # Generate indices of the batch.
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get indices for this batch.
        batch = np.array([self.indices[k] for k in index])
        
        # Sort the indices within the batch -- will speed up file access.
        batch = np.sort(batch)

        # Generate data. X is a list of features, y is a single feature (target).
        X, y = self.__get_data(batch)
        return list(X + [y])
    
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if(self.shuffle): np.random.shuffle(self.index)
    
    def __get_data(self, batch):
        X = {
            br:self.scalar_array[br][batch].to_numpy()
            for br in self.scalar_branches
            }
        X = {**X, **self.image_array[batch]} # needs Python 3.5+
        
        X = [self.scalar_array[br][batch].to_numpy() for br in self.scalar_branches]
        X += [self.image_array[batch][x] for x in self.matrix_branches]
        
        if(self.target is not None):
            y = self.scalar_array[self.target][batch].to_numpy()
        else: y = None
        return X,y
    
    def get_feature_names(self):
        return self.scalar_branches + self.matrix_branches
    
    def get_target_name(self):
        return self.target
    
    def get_feature_types(self, return_tf=True):
        types = [self.scalar_array[br][[0]].to_numpy().dtype for br in self.scalar_branches]
        types += self.image_array.get_types()
        if(return_tf): types = [tf.dtypes.as_dtype(x) for x in types]
        return types
    
    def get_target_type(self, return_tf=True):
        types = self.scalar_array[self.target][[0]].to_numpy().dtype
        if(return_tf): types = tf.dtypes.as_dtype(types)
        return types
    
    def get_feature_shapes(self):
        shapes = [(self.batch_size,) for _ in self.scalar_branches]
        shapes += [tuple([self.batch_size, *x]) for x in self.image_array.image_shapes]
        return shapes
    
    def get_target_shape(self):
        shape = (self.batch_size,)
        return shape
    
    def get_names(self):
        return tuple(self.get_feature_names() + [self.get_target_name()])
    
    def get_types(self):
        return tuple(self.get_feature_types() + [self.get_target_type()])
    
    def get_shapes(self):
        return tuple(self.get_feature_shapes() + [self.get_target_shape()])    
    
    def __call__(self):
        for idx in self.index:
            yield self.__getitem__(idx)