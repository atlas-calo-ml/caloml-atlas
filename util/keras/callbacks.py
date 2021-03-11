import tensorflow as tf
from util.keras import clr, lrlog, lr_schedule, epoch_update

# -- Define a bunch of different callbacks... --

# Checkpoint the model (by default, at every epoch).
def Checkpoint(modelfile, monitor='val_loss', save_best_only=False,save_freq='epoch'):
    if('.h5' not in modelfile): modelfile = modelfile #+ '.ckpt'
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=modelfile,
        monitor=monitor,
        verbose=0,
        save_best_only=save_best_only,
        save_weights_only=False,
        mode='auto',
        save_freq=save_freq
    )

# Log our trainng metrics (loss etc.) in a CSV file.
def Logger(modelfile, append=True):
    if('.h5' in modelfile): history_filename = '.'.join(modelfile.split('.')[:-1]) + '.csv'
    else: history_filename = modelfile + '.csv' # if using .tf format, there won't be a file extension on the string at all.    
    return tf.keras.callbacks.CSVLogger(
        filename=history_filename,
        append=append
    )

# Exponential learning rate decay.
def LrDecay(gamma=0.1):
    return lr_schedule.LearningRateSchedule(mode='exp',gamma=gamma)

# Add learning rate to the list of logged parameters.
def LrLog():
    return lrlog.LRLog()

# Modify the optimizer to use a cyclic learning rate.
def CyclicLearningRate(base_lr, max_lr, step_size=2.0e3, mode='triangular2',gamma=1., scale_fn=None, scale_mode='cycle'):
    return clr.clr(
        base_lr=base_lr, 
        max_lr=max_lr, 
        step_size=step_size, 
        mode=mode,
        gamma=gamma, 
        scale_fn=scale_fn, 
        scale_mode=scale_mode
    )

# Early stopping
def EarlyStop(monitor='val_loss', min_delta=.01, patience=2, verbose=1, restore_best_weights=True):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        restore_best_weights=restore_best_weights
    )
    

# -- ... and a function that simply returns a default set of them! --
def GetCallbacks(modelfile, append=True, use_decay=True, use_clr=False, use_stopping=True, **kwargs):
    
    checkpt = Checkpoint(modelfile, save_best_only=use_stopping)
    callbacks = [checkpt]
    
    lrl = LrLog()
    logger = Logger(modelfile,append)

    if(use_decay):
        decay = LrDecay(kwargs['gamma'])
        callbacks.append(decay)
    if(use_clr):
        CLR = CyclicLearningRate(**kwargs)
        callbacks.append(CLR)
        
    if(use_stopping):
        if('min_delta' in kwargs.keys()): min_delta = kwargs['min_delta']
        else: min_delta = 0.01
            
        if('patience' in kwargs.keys()): patience = kwargs['patience']
        else: patience = 2            
            
        early_stop = EarlyStop(
            min_delta=min_delta,
            patience=patience
        )
        callbacks.append(early_stop)

    callbacks = callbacks + [lrl, logger]
    return callbacks