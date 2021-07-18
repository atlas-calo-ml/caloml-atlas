# Some stuff for using AutoQKeras to automatically quantize models.
import pathlib, io
import numpy as np
import pandas as pd
from qkeras.autoqkeras import AutoQKeras
from util.classification.training_util import TrainNetwork
from contextlib import redirect_stdout
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects as add_qkeras_layers

def AutoQuantization(model, custom_objects, metrics, 
                     x_train, y_train,
                     x_valid, y_valid,
                     callbacks,
                     epochs, batch_size,
                     modelfile,
                     qcallbacks=None,
                     qepochs=None, qbatch_size=None,
                     qmodelfile=None,
                     run_configuration=None):
    '''
    Perform an AutoQKeras routine -- returning a trained, quantized model.
    The run configuration dictionary contains *a lot* of information,
    and this routine may be quite long and produce a lot of printouts.
    '''
    
    if(run_configuration is None):
        print('Error: No run configuration given.')
        assert(False)
        
    if(qmodelfile is None):
        qmodelfile = modelfile.replace('.h5','_q.tf')
        
    # Train the original (unquantized) model if it is not already trained.
    model, history = TrainNetwork(model,modelfile, x_train, y_train, x_valid, y_valid, callbacks=callbacks,
                                  sample_weight=None, epochs=epochs, batch_size=batch_size, verbose=1,
                                  overwriteModel=False, finishTraining=False
                                 )
    
    # If the quantized model file already exists, load it and return the model.
    if(pathlib.Path(qmodelfile).exists()):
        # Get the history file (name should follow this pattern).
        historyfile = '.'.join(qmodelfile.split('.')[:-1]) + '.csv'
        
        co = custom_objects
        add_qkeras_layers(co)
        
        print(co)
        
        # Load the model.
        qmodel = load_model(qmodelfile, custom_objects=co, compile=False) # TODO: using compile=False to avoid some issues
        
        # Load the history.
        qhistory = {}
        df = pd.read_csv(historyfile)
        for key in df.keys():
            qhistory[key] = df[key].to_numpy()
        return qmodel, qhistory, model, history # TODO: Do we want to return regular model and history too?
    
    # Prepare AutoQKeras (defining search space, establishing some baseline quantization based on run_configuration).
    autoqk = AutoQKeras(model, 
                        metrics=metrics, 
                        custom_objects=custom_objects, 
                        **run_configuration
                       )
    
    # Now perform the search. We use a larger batch size and smaller number of epochs to speed up this search part.
    # (After we get our optimal configuration, we will retrain to really squeeze out the best performance).
    
    if(qbatch_size is None): qbatch_size = int(np.minimum(10 * batch_size, 1024))
    if(qepochs is None): qepochs = int(np.maximum(epochs/10, 20))
    
    qhistory = autoqk.fit(x_train, y_train,
                          validation_data=(x_valid, y_valid),
                          batch_size=qbatch_size,
                          epochs=qepochs
    )
    
    # Once the above is completed, we need to fetch our best model. This will also send some useful information
    # on energy reduction to stdout, so we want to capture that too.
    
    f = io.StringIO()
    with redirect_stdout(f):
        qmodel = autoqk.get_best_model()
    quantization_info = f.getvalue() # this string contains info on the energy reduction
    
    # For now, we'll save the quantization info to a log file.
    quantization_log = qmodelfile.replace('.h5','.log').replace('.tf','.log')
    if('.h5' not in qmodelfile and '.tf' not in qmodelfile): 
        quantization_log = qmodelfile + '.log'
    with open(quantization_log,'w') as f:
        f.write(quantization_info)
    
    #qmodel.save_weights(qmodelfile)
    #qmodel.load_weights(modelfile)
    
    # Now that we have our best model, we can properly train it. We will use the same # of epochs and batch_size
    # as for the unquantized model, but we will optionally provide specialized callbacks (e.g. CSVLogger with a different
    # log file than the unquantized model).
    if(qcallbacks is None): qcallbacks = []
    qmodel, qhistory = TrainNetwork(qmodel,qmodelfile, x_train, y_train, x_valid, y_valid, callbacks=qcallbacks,
                                    sample_weight=None, epochs=epochs, batch_size=batch_size, verbose=1,
                                    overwriteModel=False, finishTraining=False, custom_objects=custom_objects
                                   )
    
    return qmodel, qhistory, model, history

