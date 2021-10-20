# Utilities for jet clustering, plotting jet kinematics et cetera. These are functions used in our jet-clustering workflow,
# so some many not have to do *explicitly* with jets (e.g. there might be some stuff for plotting topo-cluster predicted energies).
# Note that functions that are purely for convenience will be placed in qol_util.py.
import sys, os, glob, uuid
import ROOT as rt
import uproot as ur
import numpy as np
import subprocess as sub
from numba import jit
from util import qol_util as qu

# Fastjet setup.
def BuildFastjet(fastjet_dir=None, j=4, force=False, verbose=False):
    if(fastjet_dir is None):
        fastjet_dir = os.path.dirname(os.path.abspath(__file__)) + '/../fastjet'
        
    # Check if Fastjet is already built at destination.
    # Specifically, we will look for some Python-related files.
    if(not force):
        
        files_to_find = [
            '{}/**/site-packages/fastjet.py'.format(fastjet_dir),
            '{}/**/site-packages/_fastjet.a'.format(fastjet_dir),
            '{}/**/site-packages/_fastjet.so.0'.format(fastjet_dir)
        ]
        
        files_to_find = [glob.glob(x,recursive=True) for x in files_to_find]
        files_to_find = [len(x) for x in files_to_find]
        if(0 not in files_to_find):
            if(verbose): print('Found existing Fastjet installation with Python extension @ {}.'.format(fastjet_dir))
            return fastjet_dir
    
    # Make the Fastjet dir if it does not exist.
    try: os.makedirs(fastjet_dir)
    except: pass
    
    # Put output into log files.
    logfile = '{}/log.stdout'.format(fastjet_dir)
    errfile = '{}/log.stderr'.format(fastjet_dir)
    
    with open(logfile,'w') as f, open(errfile,'w') as g:
    
        # Fetch the Fastjet source
        fastjet_download = 'http://fastjet.fr/repo/fastjet-3.4.0.tar.gz'
        print('Downloading fastjet from {}.'.format(fastjet_download))
        sub.check_call(['wget', fastjet_download], 
                       shell=False, cwd=fastjet_dir, stdout=f, stderr=g)
        sub.check_call(['tar', 'zxvf', 'fastjet-3.4.0.tar.gz'], 
                       shell=False, cwd=fastjet_dir, stdout=f, stderr=g)
        sub.check_call(['rm', 'fastjet-3.4.0.tar.gz'], 
                       shell=False, cwd=fastjet_dir, stdout=f, stderr=g)

        source_dir  = '{}/fastjet-3.4.0'.format(fastjet_dir)
        install_dir = '{}/fastjet-install'.format(fastjet_dir)

        # Now configure. We create the python bindings.
        print('Configuring fastjet.')
        sub.check_call(['./configure', '--prefix={}'.format(install_dir), '--enable-pyext'], 
                       shell=False, cwd=source_dir, stdout=f, stderr=g)

        # Now make and install. Will skip "make check".
        print('Making fastjet.')
        sub.check_call(['make', '-j{}'.format(j)], 
                       shell=False, cwd=source_dir, stdout=f, stderr=g)
        print('Installing fastjet.')
        sub.check_call(['make', 'install'], 
                       shell=False, cwd=source_dir, stdout=f, stderr=g)
    return install_dir

# Polar to Cartesian, for circumventing TLorentzVector (etc) usage.
def Polar2Cartesian(pt,eta,phi,e):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.array([px,py,pz,e],dtype=np.dtype('f8')).T