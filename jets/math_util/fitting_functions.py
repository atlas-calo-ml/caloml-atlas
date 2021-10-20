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
    result = 2. * np.exp(x**2) * x
    result += -2. * (np.sqrt(np.pi) * np.exp(x**2) * x * spec.erf(x) + 1.) / np.sqrt(np.pi)
    return result

# Derivative of erfcxinv
def d_erfcxinv(x):
    result = 1. / d_erfcx(erfcxinv(x))
    return result

class Gaussian:
    '''
    Standard Gaussian function (normal distribution with some pre-factor).
    '''
    def __init__(self):
        self.npar = 3
        self.name = 'Gaussian'
    
    def eval(self,x,p):
        return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))
    
    def peak(self,p,dp):
        value = p[1]
        uncertainty = dp[1]
        return (value,uncertainty)
    
    # Gives FWHM
    def width(self,p,dp):
        factor = 2. * np.sqrt(2. * np.log(2.))
        value = factor * p[2]
        uncertainty = factor * dp[2]
        return (value,uncertainty)
    
    # Convenience function
    def set_parameters(self,tf1,h):
        tf1.SetParameter(0,h.GetMaximum())
        tf1.SetParameter(1,h.GetMean())
        tf1.SetParameter(2,h.GetRMS())
        return
        
class AsymmGaussian:
    '''
    Asymmetric Gaussian function -- the standard deviation can be different
    to the left and right of the mean (2 fitting parameters instead of 1).
    '''
    def __init__(self):
        self.npar = 4
        self.name = 'AsymmGaussian'
        
    def eval(self,x,p):
        if(x[0] <= p[1]):
            return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[2]))
        else:
            return p[0] * np.exp(-0.5 * np.square((x[0]-p[1])/p[3]))
        
    def peak(self,p,dp):
        value = p[1]
        uncertainty = dp[1]
        return (value,uncertainty)
    
    # Gives FWHM
    def width(self,p,dp):
        factor = 2. * np.sqrt(2. * np.log(2.))
        value = factor * (p[2] + p[3])/2.
        uncertainty = factor * np.sqrt(np.square(dp[2] / 2.) + np.square(dp[3] / 2.))
        return (value,uncertainty)
    
    # Convenience function
    def set_parameters(self,tf1,h):
        tf1.SetParameter(0,h.GetMaximum())
        tf1.SetParameter(1,h.GetMean())
        tf1.SetParameter(2,h.GetRMS())
        tf1.SetParameter(3,h.GetRMS())
        return

class AsymmGaussianLike:
    '''
    Asymmetric Gaussian function -- the standard deviation can be different
    to the left and right of the mean (2 fitting parameters instead of 1).
    '''
    def __init__(self):
        self.npar = 6
        self.name = 'AsymmGenGaussian'
        
    def eval(self,x,p):
        if(x[0] <= p[1]):
            return p[0] * np.exp(-0.5 * np.power( np.abs((x[0]-p[1])/p[2]) , p[4]))
        else:
            return p[0] * np.exp(-0.5 * np.power( np.abs((x[0]-p[1])/p[3]) , p[5]))

    def peak(self,p,dp):
        value = p[1]
        uncertainty = dp[1]
        return (value,uncertainty)
    
    # Gives FWHM
    def width(self,p,dp):
        ep = np.log(2.)
        value =  2. * np.power(ep, 1./p[4]) * p[2]
        value += 2. * np.power(ep, 1./p[5]) * p[3]
        uncertainty =  np.power(ep, 2./p[4]) * (np.square(dp[2]) + np.square(np.log(ep) / np.square(p[4])) * np.square(dp[4]))
        uncertainty += np.power(ep, 2./p[5]) * (np.square(dp[3]) + np.square(np.log(ep) / np.square(p[5])) * np.square(dp[5]))
        return (value,uncertainty)
    
    # Convenience function
    def set_parameters(self,tf1,h):
        tf1.SetParameter(0,h.GetMaximum())
        tf1.SetParameter(1,h.GetMean())
        tf1.SetParameter(2,h.GetRMS())
        tf1.SetParameter(3,h.GetRMS())
        tf1.SetParameter(4,2.)
        tf1.SetParameter(5,1.9)
        
        tf1.FixParameter(4,2.)
        
        tf1.SetParLimits(2,0., 2.*h.GetRMS())
        tf1.SetParLimits(3,0., 2.*h.GetRMS())
        tf1.SetParLimits(5,0.5,5.)
        return
    
    
    
# class AsymmGaussianLike:
#     '''
#     Like the asymmetric Gaussian, but now the exponents are also fitting parameters.
#     This allows not only for different widths but for different curvatures.
#     '''
#     def __init__(self):
#         self.npar = 4
#         self.name = 'AsymmGaussianLike'
        
#     def eval(self,x,p):
#         if(x[0] <= p[1]):
#             return p[0] * np.exp(-0.5 * np.power((x[0]-p[1])/p[2]) , 2.)
#             #return p[0] * np.exp(-1. * np.power( np.abs((x[0]-p[1])/p[2]), p[4])) # note the factor of -1 instead of -1/2 in exponent, this is a bit of an arbitrary choice
#         else:
#             return p[0] * np.exp(-0.5 * np.power((x[0]-p[1])/p[3]) , 2.)
#             #return p[0] * np.exp(-1. * np.power(( np.abs(x[0]-p[1])/p[3]), p[5]))
        
#     def peak(self,p,dp):
#         value = p[1]
#         uncertainty = dp[1]
#         return (value,uncertainty)
    
#     # Gives FWHM
#     def width(self,p,dp):
#         ep = np.log(2.)
# #         value =  2. * np.power(ep, 1./p[4]) * p[2]
# #         value += 2. * np.power(ep, 1./p[5]) * p[3]
# #         uncertainty =  np.power(ep, 2./p[4]) * (np.square(dp[2]) + np.square(np.log(ep) / np.square(p[4])) * np.square(dp[4]))
# #         uncertainty += np.power(ep, 2./p[5]) * (np.square(dp[3]) + np.square(np.log(ep) / np.square(p[5])) * np.square(dp[5]))
#         value = 0.
#         uncertainty = 0.
#         return (value,uncertainty)
    
#     # Convenience function
#     def set_parameters(self,tf1,h):
#         tf1.SetParameter(0,h.GetMaximum())
#         tf1.SetParameter(1,h.GetMean())
#         tf1.SetParameter(2,h.GetRMS())
#         tf1.SetParameter(3,h.GetRMS())
# #         tf1.SetParameter(4,2.)
# #         tf1.SetParameter(5,2.)
# #         tf1.SetParLimits(4,1.9,2.1)
# #         tf1.SetParLimits(5,1.9,2.1)

#         return

class ExGaussian:
    def __init__(self):
        self.npar = 4
        self.name = 'ExGaussian'
        
    def eval(self,x,p):
        ls2 = p[3] * p[2] * p[2]
        result = p[0] * np.exp(p[3]/2. * (2. * p[1] + ls2 - 2. * x[0]))
        result *= 1. - spec.erf((p[1] + ls2 - x[0])/(np.sqrt(2) * p[2]))
        return result  
    
    def peak(self,p,dp):
        
        # convenience definitions
        sqpi = np.sqrt(np.pi)
        earg = np.sqrt(2. / np.pi) / (p[2] * p[3])
        
        
        value = p[1] - np.sign(p[3]) * np.sqrt(2) * p[2] * erfcxinv(np.sqrt(2./np.pi) /(p[2] * p[3])) + p[3] * np.square(p[2])
        value = value.item()
        # Somewhat complex uncertainty expression.
        uncertainty = np.array([
            np.atleast_1d(0.), # dpeak / dp[0]
            np.atleast_1d(1.), # dpeak / dp[1]
            -np.sqrt(2) * erfcxinv(earg) + 2./sqpi /(p[2] * p[3]) * d_erfcxinv(earg) + 2. * p[2] * p[3], # dpeak / dp[2]
            2./sqpi / np.square(p[3]) * d_erfcxinv(earg) + np.square(p[2]) # dpeak / dp[3]
        ])
        
        uncertainty = uncertainty.flatten()
        uncertainty = np.sqrt(np.dot(np.square(uncertainty),np.square(dp)))
        return (value,uncertainty)
    
    # TODO: Need to find FWHM, will probably assume it scales with sigma (and maybe lambda)
    
    # An approximation of the FWHM, which we've determined empirically (from some plotting/testing).
    # This should at least work within the range of typical parameters.
    def width(self,p,dp):
        # FWHM = a * sigma^b + c * lambda^d
        a = 2.27
        b = 1.03
        c = 6.81e-1
        d = -1.03
        value = a * np.power(p[2],b) + c * np.power(p[3],d)
        uncertainty = np.sqrt(np.square( a * b * np.power(p[2], b - 1.) * dp[2]) + np.square(c * d * np.power(p[3], d - 1.) *  dp[3]))
        return (value,uncertainty)    
    
    # Convenience function
    def set_parameters(self,tf1,h):
        tf1.SetParameter(0,h.GetMaximum())
        tf1.SetParameter(1,h.GetMean())
        tf1.SetParameter(2,h.GetRMS())
        tf1.SetParameter(3,3.)
        tf1.SetParLimits(3,2.,5.) # in practice this limit might be helpful
        return    
    

    
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
    
    
    