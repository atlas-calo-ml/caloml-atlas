import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from util import plot_util  as pu
from util import qol_util as qu
import itertools

# def EnergyMapping(x, b=0.):
#     return np.sign(x) * np.log(np.abs(x) + b)

# def EnergyMappingInverse(x, b=0.):
#     return np.sign(x) * (np.exp(np.abs(x)) - b)

class SimpleLogMapping:
    
    def __init__(self,b=0.,m=0.):
        self.b = b # unused
        self.m = m # unused

    def Forward(self,x):
        return np.log(x)
        
    def Inverse(self,x):
        return np.exp(x)
        
class LogMapping:
    def __init__(self,b=1.,m=0.):
        self.b = b
        self.m = m # unused
        
    #@jit
    def Forward(self,x):
        return np.sign(x) * np.log(np.abs(x) + self.b)
    #@jit    
    def Inverse(self,x):
        return np.sign(x) * (np.exp(np.abs(x)) - self.b)

class LinLogMapping:
    def __init__(self,b=1.,m=1.):
        self.b = b
        self.m = m
    
    #@jit
    def Forward(self, x):
        result = np.zeros(x.shape, dtype=np.dtype('f8'))
    
        for i in range(len(x)):
            if(x[i] <= self.b): result[i] = self.m * x[i]
            else: result[i] = np.log(self.m * (x[i] - self.b) + 1.) + self.m * self.b
        return result

    #@jit
    def Inverse(self, x):
        result = np.zeros(x.shape, dtype=np.dtype('f8'))
        mb = self.m * self.b
        for i in range(len(x)):
            if(x[i] <= mb): result[i] = x[i] / self.m
            else: result[i] = (np.exp(x[i] - mb) - 1.) / self.m + self.b
        return result

def MapStabilityTest(mapping_func, b_vals=[0.,.1,.5,1.,1.0e14], m_vals=[1.], x=np.linspace(0.001,4.,1000), ps=qu.PlotStyle('dark'),savedir='',legend_size=-1):
    
    mb_combos = list(itertools.product(b_vals,m_vals))
    
    if(m_vals == [1.]):
        forward_labels = ['f(x), b={:.1e}'.format(mb[0])    for mb in mb_combos]
        reverse_labels = ['g(f(x)), b={:.1e}'.format(mb[0]) for mb in mb_combos]
    else:
        forward_labels = ['f(x), b={:.1e}, m={:.1e}'.format(mb[0],mb[1])    for mb in mb_combos]
        reverse_labels = ['g(f(x)), b={:.1e}, m={:.1e}'.format(mb[0],mb[1]) for mb in mb_combos]

    forward = [mapping_func(b,m).Forward(x) for (b,m) in mb_combos]
    reverse = [mapping_func(mb_combos[i][0],mb_combos[i][1]).Inverse(forward[i]) for i in range(len(forward))]
    
    #reverse = [mapping_func(b,m).Inverse(forward) for (b,m) in mb_combos]

    fig,ax = plt.subplots(1,2,figsize=(16,6))
    y_min, y_max = (np.min(x),np.max(x))
    
    pu.multiplot_common(ax[0], x, forward, forward_labels, y_min=y_min, y_max=y_max, xlabel='x', ylabel='y', title='Forward Mapping', ps=ps)
    pu.multiplot_common(ax[1], x, reverse, reverse_labels, y_min=y_min, y_max=y_max, xlabel='x', ylabel='y', title='Reverse Mapping', ps=ps)
    
    if(legend_size > 0):plt.rc('legend',fontsize=legend_size)
    plt.show()
    
    savename = 'mapping_test.png'
    if(savedir != ''):
        savename = savedir + '/' + savename
    fig.savefig(savename,transparent=True)
    return