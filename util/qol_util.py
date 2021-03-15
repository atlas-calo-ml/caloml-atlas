# Just some simple, quality-of-life functions. Nothing very fancy.

import numpy as np
import matplotlib.pyplot as plt
import ROOT as rt
import sys, os, uuid

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

# Plot display/adjustments.
# This sets a bunch of colors, for both PyROOT and matplotlib.
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
            self.curve = rt.TColor.GetColor(249,105,4)
            
            self.text_plt = 'xkcd:white'
            self.canv_plt = '#222222' # dark grey
            self.main_plt = '#34a5da' # light blue
            self.grid_plt = '0.65'
            self.curve_plt = '#f96904' # orange
            
        # list of matplotlib colors
        self.colors = [
            'xkcd:medium purple', # first pass thrpugh rainbow
            'xkcd:periwinkle blue', 
            'xkcd:aqua blue',
            'xkcd:electric lime',
            'xkcd:kelly green',
            'xkcd:tangerine',
            'xkcd:wheat',
            'xkcd:bordeaux',
            'xkcd:bright red',
            'xkcd:baby purple', # second pass through rainbow
            'xkcd:dark teal',
            'xkcd:true blue',
            'xkcd:very light green',
            'xkcd:macaroni and cheese',
            'xkcd:burnt orange',
            'xkcd:brick red',
            'xkcd:salmon'
         ]
        #self.colors.reverse() # put the reds first -- purple can be comparitively hard to see on dark background
        
        # list of matplotlib linestyles
        self.linestyles = [
            '-',
            ':',
            '-.'
        ]
            
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
        
    def SetStylePlt(self, ax):
        
        # canvas color
        ax.set_facecolor(self.canv_plt)
    
        # tick colors (marks, then tick labels)
        ax.tick_params(axis='both',colors=self.main_plt)
        plt.setp(ax.get_xticklabels(), color=self.text_plt)
        plt.setp(ax.get_yticklabels(), color=self.text_plt)
    
        # axis spines
        for spine in ['bottom','top','left','right']:
            ax.spines[spine].set_color(self.main_plt)

        # axis titles
        ax.xaxis.label.set_color(self.text_plt)
        ax.yaxis.label.set_color(self.text_plt)
    
        # plot title
        ax.title.set_color(self.text_plt)
        
        # grid color
        ax.grid()


# Setting a histogram's line and fill color in one go
def SetColor(hist, color, alpha = 1., style=0):
    hist.SetFillColorAlpha(color, alpha)
    hist.SetLineColor(color)
    if(style != 0): hist.SetFillStyle(style)
    return

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


# Saving subplots to individual files. Can't believe this isn't built-in to matplotlib.
def SaveSubplots(fig, axes, names, extensions = ['png'], savedir='', ps=PlotStyle('dark')):
    assert(len(axes.flatten()) == len(names))
    for i in range(len(names)):
        bbox = axes.flatten()[i].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        
        for ext in extensions:
            savename = names[i] + '.' + ext
            if(savedir != ''): savename = savedir + '/' + savename
            plt.savefig(savename,bbox_inches=bbox, facecolor=ps.canv_plt)
    return

# Hiding print statements.
# For hiding print statements. TODO: Not working for suppressing fastjet printouts.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
   