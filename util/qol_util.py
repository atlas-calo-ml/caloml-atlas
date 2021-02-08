# Just some simple, quality-of-life functions. Nothing very fancy.

import numpy as np
import ROOT as rt
import sys, os, uuid

# 1) Progress bars.

# Print iterations progress.
# Adapted from https://stackoverflow.com/a/34325723.
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
# Progress bar with color.
def printProgressBarColor (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    fill_prefix = '\33[31m'
    fill_suffix = '\033[0m'
    prog = iteration/total
    if(prog > 0.33 and prog <= 0.67): fill_prefix = '\33[33m'
    elif(prog > 0.67): fill_prefix = '\33[32m'
    fill = fill_prefix + fill + fill_suffix
    printProgressBar(iteration, total, prefix = prefix, suffix = suffix, decimals = decimals, length = length, fill = fill, printEnd = printEnd)
    return

#2) Plot display/adjustments.
class PlotStyle:
    def __init__(self, mode = 'dark'):
        if(mode != 'dark'): mode = 'light'
            
        if(mode == 'light'):
            self.main  = rt.kBlack
            self.canv  = rt.kWhite
            self.text  = rt.kBlack
            self.curve = rt.kBlue
            
            self.text_plt = 'xkcd:black'
            self.canv_plt = 'xkcd:white'
            self.main_plt = 'xkcd:black'
            self.grid_plt = '0.65'
            self.curve_plt = 'xkcd:blue'
            
        elif(mode == 'dark'):
            self.main  = rt.TColor.GetColor(52,165,218)
            self.canv  = rt.TColor.GetColor(34,34,34)
            self.text  = rt.kWhite
            self.curve = rt.TColor.GetColor(231,114,61)
            
            self.text_plt = 'xkcd:white'
            self.canv_plt = '#222222'
            self.main_plt = 'xkcd:black'
            self.grid_plt = '0.65'
            self.curve_plt = '#34a5da'
            
    def SetStyle(self):
        rt.gStyle.SetAxisColor(self.main,'xyz')
        rt.gStyle.SetGridColor(self.main)
        rt.gStyle.SetLineColor(self.main)
        rt.gStyle.SetFrameLineColor(self.main)
        
        rt.gStyle.SetPadColor(self.canv)
        rt.gStyle.SetCanvasColor(self.canv)
        rt.gStyle.SetLegendFillColor(self.canv)
        
        rt.gStyle.SetTitleTextColor(self.text)
        rt.gStyle.SetTitleColor(self.text, 'xyz')
        rt.gStyle.SetLabelColor(self.text, 'xyz')
        rt.gStyle.SetStatTextColor(self.text)
        rt.gStyle.SetTextColor(self.text)


# Setting a histogram's line and fill color in one go
def SetColor(hist, color, alpha = 0.5):
    hist.SetLineColor(color)
    hist.SetFillColorAlpha(color, alpha)

def RN():
    return str(uuid.uuid4())
    
# Plotting groups of histograms together, in separate tiles.
def DrawSet(hists, logx=False, logy=True, paves = 0):
    nx = 2
    l = len(hists.keys())
    ny = int(np.ceil(l / nx))
    canvas = rt.TCanvas(RN(), RN(), 600 * nx, 450 * ny)
    canvas.Divide(nx, ny)
    for i, hist in enumerate(hists.values()):
        canvas.cd(i+1)
        hist.Draw('HIST')
        if(logx):
            rt.gPad.SetLogx()
            hist.GetXaxis().SetRangeUser(1.0e-0, hist.GetXaxis().GetBinUpEdge(hist.GetXaxis().GetLast()))
        if(logy): 
            rt.gPad.SetLogy()
            hist.SetMinimum(5.0e-1)
        else:
            hist.SetMinimum(0.)
        if(paves != 0):
            for pave in paves: pave.Draw()
    return canvas

# 3) Hiding print statements.
# For hiding print statements. TODO: Not working for suppressing fastjet printouts.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# -- Unused functions / funcs for deprecated workflows below --

# Some ROOT/numpy stuff
# Converting from ROOT type names to leaflist decorators.
# Vector decorator will not work, but gives a sensible string
# telling us the depth (how many vectors).
def RTypeConversion(type_string):
    if(type_string == 'Short_t' or type_string == 'short'):    return 'S'
    elif(type_string == 'Int_t' or type_string == 'int'):    return 'I'
    elif(type_string == 'Float_t' or type_string == 'float'):  return 'F'
    elif(type_string == 'Double_t' or type_string == 'double'): return 'D'
    elif('vector' in type_string): # special case
#         type_substring = '<'.join(type_string.split('<')[1:])
#         type_substring = '>'.join(type_substring.split('>')[:-1])
#         type_substring = RTypeConversion(type_substring)
#         return 'v_' + type_substring
        return type_string
    else: return '?'

def GetShape(shape_string):
    dims = shape_string.replace('[',' ').replace(']', ' ').split()
    return tuple([int(x) for x in dims])

def RType2NType(type_string):
    if(type_string == 'S'):   return np.dtype('i2')
    elif(type_string == 'I'): return np.dtype('i4')
    elif(type_string == 'L'): return np.dtype('i8')
    elif(type_string == 'F'): return np.dtype('f4')
    elif(type_string == 'D'): return np.dtype('f8')
    else: raise ValueError('Input not understood.')

        