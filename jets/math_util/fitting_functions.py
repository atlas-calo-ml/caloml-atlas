# A bunch of functions for fitting histograms.
# TODO: Would be nice to turn these into classes for more advanced usage.
import numpy as np
import scipy.special as spec
from math_util.erfcxinv import EvaluatePrecise
import subprocess as sub

# Inverse scaled complementary error function (erfcx inverse)
def erfcxinv(x):
    filename = 'erfcxinv.h5'
    result = EvaluatePrecise(x, filename)
    sub.check_call(['rm',filename])
    return result

# Derivative of erfcx
def d_erfcx(x):
    sqpi = np.sqrt(np.pi)
    ex2x = np.exp(np.square(x)) * x
    result = -2. * (sqpi * ex2x * spec.erf(x) + 1.) / sqpi
    result += 2. * ex2x
    return result

# Derivative of erfcxinv
def d_erfcxinv(x):
    result = 1. / d_erfcx(erfcxinv(x))
    return result

class Gaussian:
    
    def eval(self,x,p):
        return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))
    
    def peak(self,p,dp):
        value = p[1]
        uncertainty = dp[1]
        return (value,uncertainty)
    
    def width(self,p,dp):
        value = p[2]
        uncertainty = dp[2]
        return (value,uncertainty)
    
class AsymmGaussian:
    
    def eval(self,x,p):
        if(x[0] <= p[1]):
            return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))
        else:
            return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[3]))
        
    def peak(self,p,dp):
        value = p[1]
        uncertainty = dp[1]
        return (value,uncertainty)
    
    def width(self,p,dp):
        value = (p[2] + p[3])/2.
        uncertainty = np.sqrt(np.square(dp[2] / 2.) + np.square(dp[3] / 2.))
        return (value,uncertainty)
        
class ExGaussian:
    
    def eval(self,x,p):
        ls2 = p[3] * p[2] * p[2]
        result = p[0] * np.exp(p[3]/2. * (2. * p[1] + ls2 - 2. * x[0]))
        result *= 1. - spec.erf((p[1] + ls2 - x[0])/(np.sqrt(2) * p[2]))
        return result  
    
    def peak(self,p,dp):
        sqpi = np.sqrt(np.pi)
        
        value = p[1] - np.sign(p[3]) * np.sqrt(2) * p[2] * erfcxinv(np.sqrt(2./np.pi) /(p[2] * p[3]))
        
        # Somewhat complex uncertainty expression.
        uncertainty = np.array([
            1., # dpeak / dp[1]
            2./(sqpi * np.square(p[3]))        * d_erfcxinv(np.sqrt(2./np.pi) /(p[2] * p[3])), # dpeak / dp[2]
            2./(sqpi * np.square(p[2]) * p[3]) * d_erfcxinv(np.sqrt(2./np.pi) /(p[2] * p[3])), # dpeak / dp[3]
        ])
        uncertainty = np.sqrt(np.dot(np.square(uncertainty),np.square(dp)))
        return (value,uncertainty)
    
    
# def Gaussian(x,p):
#     '''
#     A basic Gaussian function.
#     '''
#     return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))


# def AsymmGaussian(x,p):
#     '''
#     A basic asymmetric Gaussian -- the sigma can be different to the left and right of the mean.
#     '''
#     if(x[0] <= p[1]):
#         return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))
#     else:
#         return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[3]))
    

# def ExGaussian(x,p):
#     '''
#     Exponentially-modified Gaussian function.
#     '''
#     ls2 = p[3] * p[2] * p[2]
#     result = p[0] * np.exp(p[3]/2. * (2. * p[1] + ls2 - 2. * x[0]))
#     result *= 1. - spec.erf((p[1] + ls2 - x[0])/(np.sqrt(2) * p[2]))
#     return result

# def LogLogistic(x,p):
#     '''
#     Log-logistic (Fisk) distribution.
#     '''
#     xa = x[0]/p[1]
#     result = p[0] * (p[1]/p[1]) * np.power(xa,p[2]-1)
#     result /= np.square(1. + np.power(xa,p[2]))
#     return result

# def Frechet(x,p):
#     '''
#     Frechet (inverse Weibull) distribution.
#     '''
#     return p[0] * np.power((x-p[1])/p[2],-1.-p[3]) *  np.exp(-1. * np.power((x[0]-p[1])/p[2],-p[3]))
    
    
    