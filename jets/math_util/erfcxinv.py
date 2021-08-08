import sys, os, pathlib, uuid
import subprocess as sub
import numpy as np
from numba import jit
import scipy.special as spec
import h5py as h5
import matplotlib.pyplot as plt

@jit
def GetLowerNeighbor(array,val):
    idx = -1
    N = array.shape[0]
    for i in range(N):
        if(array[i] > val):
            idx = i - 1
            break
    return idx

def CreateData(x_range=(0.,2.),npoints = 1000000,filename='erfcxinv.h5'):
    # Create a table of values for erfc(x), so that we can numerically estimate the inverse.
    # We're only interested in a limited range of values.
    x = np.flip(np.linspace(*x_range,npoints)) # function monotonically decreases, we want f(x) to increase with index, thus perform flip
    y = spec.erfcx(x)
    result = np.column_stack((y,x))
    f = h5.File(filename,'w')
    dset = f.create_dataset('erfcxinv',data=result,compression='gzip',compression_opts = 7)
    f.close()
    
# Given x, estimate erfcxinv(x) using known values and linear interpolation.
def Evaluate(x,filename='erfcxinv.h5',x_range=(0.,2.),npoints=1000000):
    
    if(not pathlib.Path(filename).exists()):
        CreateData(filename=filename,x_range=x_range,npoints=npoints)
        
    f = h5.File(filename,'r')
    data = f['erfcxinv'][:]
    f.close()
        
    y = np.zeros(x.shape)
    # evaluate each datapoint (assuming list/ndarray for now)
    for i,x_val in enumerate(x):
        # Find nearest two points, do linear interpolation.
        # The erfcx function is monotonically decreasing.
        idx_low = GetLowerNeighbor(data[:,0],x_val)
        idx_high = idx_low + 1
        if(idx_low < 0.): continue
        xi = data[idx_low,0]
        xf = data[idx_high,0]
        yi = data[idx_low,1]
        yf = data[idx_high,1]        
        y[i] = np.interp(x_val, (xi,xf), (yi,yf))
    return y

# Given x, estimate erfcxinv(x) using "clever" interpolation.
def EvaluatePrecise(x,filename='erfcxinv.h5', tolerance=0.001, nsteps=100, x_range=(0.,2.), npoints=10):
    x = np.atleast_1d(x)
    temp_files = []
        
    if(not pathlib.Path(filename).exists()):
        CreateData(filename=filename,x_range=x_range,npoints=npoints)
        
    f = h5.File(filename,'r')
    data = f['erfcxinv'][:]
    f.close()
        
    y = np.zeros(x.shape)
    for i,x_val in enumerate(x):
        # Find nearest two points in our lookup table.
        idx_low = GetLowerNeighbor(data[:,0],x_val)
        idx_high = idx_low + 1
        if(idx_low < 0.): continue
        xi = data[idx_low,0]
        xf = data[idx_high,0]
        yi = data[idx_low,1]
        yf = data[idx_high,1]
                
        step_size = (xf - xi) / nsteps
        
        y_val = np.interp(x_val, (xi,xf), (yi,yf))
        
        # Determine "error" by calculating erfcx(y) (?= x).
        x_test = spec.erfcx(y_val)
        error = np.atleast_1d(np.abs(x_val - x_test) / x_test)
        
        while(error > tolerance):
            tmp_file = str(uuid.uuid4()) + '.h5'
            temp_files.append(tmp_file)
            y_val = EvaluatePrecise(np.array([x_val]),tmp_file,tolerance,nsteps,x_range=(yf,yi),npoints=npoints)
            x_test = spec.erfcx(y_val)
            error = np.abs(x_val - x_test) / x_test
            
            error_sign = np.sign(x_val - x_test)
            if(error_sign < 0.):
                break
        y[i] = y_val
        
    # Delete any temporary files we have accrued.
    if(len(temp_files) > 0):
        command = ['rm'] + temp_files
        sub.check_call(command)
    return y


