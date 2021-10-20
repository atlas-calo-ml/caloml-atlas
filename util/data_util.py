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
                 image_shapes = None,
                 flatten = False):
        
        self.chain = rt.TChain(tree_name)
        for rfile in root_files: self.chain.Add(rfile)
        self.branches = image_branches
        
        self.flatten = flatten
        
        if(image_shapes is None): 
            self.image_shapes = [(mu.cell_meta[x]['len_eta'],mu.cell_meta[x]['len_phi']) for x in self.branches]
            if(self.flatten): self.image_shapes = [((x[0] * x[1]),) for x in self.image_shapes] # TODO: Is this safe? Will this cause problems w/ reading from the TTree?
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
        #if(self.flatten): return {key:val.flatten() for key,val in self.read_buffer.items()}
        return self.read_buffer
    
    def get_item_many(self,indices):
        images = {}
        for i,idx in enumerate(indices):
            buff = self.__getitem__(idx)
            if(i == 0): 
                images = buff.copy()
                # Reshape to +1 dim (1st dim will be idx of image)
                for br in self.branches:
                    images[br] = images[br][None,...]                
            else:
                for br in self.branches:
                    images[br] = np.concatenate((images[br],buff[br][None,...]),axis=0)
        #if(self.flatten): return {key:val.reshape((val.shape[0],-1)) for key,val in self.read_buffer.items()}
        return images
    
    def get_types(self, return_dict=False):
        if(return_dict): return {key:val.dtype for key,val in self.read_buffer.items()}
        return [x.dtype for x in self.read_buffer.values()]
        
# Data generator for our MLTree data.

class MLTreeV1DataGen(tf.keras.utils.Sequence):
    '''
    Data generator for our MLTree data.
    If `root_files` is given as a list or a glob-compatible pattern string,
    the `target` argument must be specified. Alternatively, if `root_files` is given
    as a dictionary of lists or glob-compatible pattern strings, then the `target` argument
    will be ignored and the dictionary keys will be treated as classification labels
    (which will be treated as the target).
    '''
    
    def __init__(self,
                 root_files,
                 tree_name,
                 scalar_branches,
                 matrix_branches = list(mu.cell_meta.keys()),
                 target=None,
                 batch_size=200,
                 shuffle=True, # TODO: Turning off shuffle caused some problems in simple tests, when retrieving data. How?
                 step_size=None,
                 flatten_images=False,
                 key_map=None):
        
        # Deal with the case of a dictionary input -- this means that the targets will be the
        # categories specified by the dictionary keys. We will need to keep track of which target
        # value each individual file is associated with, so that we can ultimately determine the
        # target value for every index in our (unshuffled) list of events.
        if(type(root_files) == dict):
            self.external_classification = True
            if(target is not None):
                print('Warning: target is set to {}, but ROOT files have been passed as a dictionary -> target will be ignored, using dictionary keys as classification labels.'.format(target))
            
            self.root_files = []
            keys = list(root_files.keys())
            keys.sort()
            nlabels = len(keys)
            self.external_classification_nclasses = nlabels

            nentries_dict = {}
            classes_dict = {}
            
            for i,key in enumerate(keys):
                rfiles = root_files[key]
                if(type(rfiles) != list): rfiles = glob.glob(rfiles,recursive=True)                    
                for rfile in rfiles:
                    with ur.open(rfile, cache=None, array_cache=None)[tree_name] as tree:
                        nentries_dict[rfile] = tree.num_entries
                        classes_dict[rfile] = i  
                self.root_files += rfiles
            self.root_files.sort()
                
            # At this point, we know how many events we have for every file, and which classification (number)
            # each file corresponds with. Thus we can determine the event index boundaries at which the classification
            # scores change -- and from this, we can determine the classification score of each event without explicitly saving
            # the score per event. In terms of memory usage, this will scale more nicely than explicitly saving all those scores.
            index_score_boundaries = {} # key is upper bound of index range (inclusive!), value is classification value
            nentries = 0
            for rfile in self.root_files:
                nentries += nentries_dict[rfile]
                index_score_boundaries[nentries-1] = classes_dict[rfile]
            self.index_score_boundaries = index_score_boundaries
            
        else:
            self.external_classification = False
            self.external_classification_nclasses = None
            self.index_score_boundaries = None
            if(type(root_files) == list): 
                self.root_files = root_files
            else:
                self.root_files = glob.glob(root_files,recursive=True)
            self.root_files.sort()
        
        if(step_size is None):
            self.step_size = '{} MB'.format(batch_size) # TODO: Is this reasonable?
        else: 
            self.step_size = step_size
        
        self.tree_name = tree_name
        self.scalar_branches = scalar_branches # We will create a lazy array for these, as it performs well.
        self.matrix_branches = matrix_branches # These will only be handled when fetching data! Not using lazy array (too slow).
 
        self.target = target
    
        # Quick hack for the case of external classification, in which case the target is redundant
        if(self.external_classification): self.target = self.scalar_branches[0]
    
        if(self.target is not None): 
            assert(self.target in self.scalar_branches)
            
        self.batch_size = batch_size
        self.shuffle = shuffle
                
        if(self.scalar_branches is None): filter_func = lambda x: x.name not in list(mu.cell_meta.keys())
        else: filter_func = lambda x: x.name in self.scalar_branches
        self.scalar_array = ur.lazy(files=[':'.join((x,self.tree_name)) for x in self.root_files],
                            filter_branch = filter_func,
                            step_size = self.step_size
                           )
        
        # Now remove the target from scalar_branches, so that it is not included among features.
        self.scalar_branches = [x for x in self.scalar_branches if x != self.target]
        
        self.branches = self.scalar_branches + self.matrix_branches
        self.key_map = key_map # optionally remap data keys (e.g. "EMB1" -> "input") for access -- this is useful if network assumes tensors have certain names that differ from actual branch names
        if(self.key_map is None):
            self.key_map = {x:x for x in self.branches}
        else:
            for x in self.branches:
                if(x not in self.key_map.keys()): self.key_map[x] = x
        
        self.image_array = ROOTImageArray(root_files = self.root_files,
                                          tree_name = self.tree_name,
                                          image_branches = self.matrix_branches,
                                          flatten = flatten_images
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
        
        # If doing classification using external labels, we need to fetch y differently.
        # We use "one-hot encoding" for y in this case.
        # See: https://stackoverflow.com/a/29831596/14765225 for one-hot encoding below.
        if(self.external_classification):
#             print('Doing thing!')
#             for key in list(self.index_score_boundaries.keys()):
#                 print('\t',key)
            boundaries = np.array(list(self.index_score_boundaries.keys()),dtype=np.dtype('i8'))
            labels = np.array(list(self.index_score_boundaries.values()),dtype=np.dtype('i8'))            
            y1 = np.array([labels[int(np.argmax(boundaries > idx - 1))] for idx in batch],dtype=np.dtype('i8'))
            y = np.zeros((y1.size, y1.max()+1))
            y[np.arange(y1.size),y1] = 1
            
#         # Optionally remap keys in X.
#         if(self.key_map is not None):
#             for old_key,new_key in self.key_map.items():
#                 if(new_key in X.keys()): continue # TODO: Handle this issue during initialization
#                 if(old_key in X.keys()):
#                     X[new_key] = X.pop[old_key]

        return X, y
    
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if(self.shuffle): np.random.shuffle(self.index)
    
    def __get_data(self, batch):
        X = {
            br:self.scalar_array[br][batch].to_numpy()
            for br in self.scalar_branches
            }
        X = {**X, **self.image_array[batch]} # needs Python 3.5+
                
        if(self.target is not None):
            y = self.scalar_array[self.target][batch].to_numpy()
        else: y = None
        return X,y
    
    def get_feature_names(self):
        names = list(self.key_map.values())
        return names
        #return self.branches
    
    def get_target_name(self):
        return self.target
    
    def get_feature_types(self, return_tf=True):
        types = {br: self.scalar_array[br][[0]].to_numpy().dtype for br in self.scalar_branches}
        types = {**types, **self.image_array.get_types(return_dict=True)}
        if(return_tf): types = {key:tf.dtypes.as_dtype(val) for key,val in types.items()}
        return types
    
    def get_target_type(self, return_tf=True):
        types = self.scalar_array[self.target][[0]].to_numpy().dtype
        if(return_tf): types = tf.dtypes.as_dtype(types)
        return types
    
    def get_feature_shapes(self):
        shapes = {br:(self.batch_size,) for br in self.scalar_branches}
        matrix_shapes = {self.matrix_branches[i] : tuple([self.batch_size,*self.image_array.image_shapes[i]]) 
                         for i in range(len(self.matrix_branches))
                        }
        shapes = {**shapes, **matrix_shapes}
        return shapes
    
    # TODO: This is a bit hacky, there is probably a more elegant way to do this.
    def get_target_shape(self):
        if(self.external_classification is not None):
            shape = (self.batch_size,self.external_classification_nclasses)
        else:
            shape = (self.batch_size,)
        return shape
    
    def get_names(self):
        return tuple(self.get_feature_names() + [self.get_target_name()])
    
    def get_types(self):
        return self.get_feature_types(),self.get_target_type()
    
    def get_shapes(self):
        return self.get_feature_shapes(),self.get_target_shape()
    
    def get_feature_signatures(self):
#         names = self.get_feature_names()
        shapes = self.get_feature_shapes()
        types = self.get_feature_types()
        sig = {key: tf.TensorSpec(shape=val, dtype=types[key], name=self.key_map[key]) for key,val in shapes.items()}
        return sig
    
    def get_target_signature(self):
        shapes = self.get_target_shape()
        types = self.get_target_type()
        names = self.get_target_name()
        sig = tf.TensorSpec(shape=shapes, dtype=types, name=names)
        return sig
    
    def get_signatures(self):
        return self.get_feature_signatures(),self.get_target_signature()
    
    def __call__(self):
        for idx in self.index:
            yield self.__getitem__(idx)
            
    
def MLTreeV1Dataset(root_files,tree_name,scalar_branches,matrix_branches = list(mu.cell_meta.keys()),
                    target=None,batch_size=200,shuffle=True,step_size=None,prefetch=True,flatten_images=False,key_map=None):
        
    generator = MLTreeV1DataGen(root_files,
                                tree_name=tree_name,
                                scalar_branches=scalar_branches,
                                matrix_branches=matrix_branches,
                                target=target,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                step_size=step_size,
                                flatten_images=flatten_images,
                                key_map=key_map)
    
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: generator, # TODO: why does "lambda: generator" work, but just "generator" doesn't?
        output_signature = generator.get_signatures()
    )    
    if(prefetch): dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # this might speed things up
        
    # Even though TF knows the length of the data when we fit by directly passing the generator, the length of this dataset
    # will still be unknown (see https://discuss.tensorflow.org/t/typeerror-dataset-length-is-unknown-tensorflow/948/2).
    # This seems like a bug/oversight, but we can easily fix it.
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(generator)))
    return dataset