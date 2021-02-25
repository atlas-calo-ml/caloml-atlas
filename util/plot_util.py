#TODO: Consolidate functions. There's lots of redundancy here (e.g. roc_plot() is just a generic plot function).

import matplotlib.font_manager
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
from util import qol_util as qu

# set plotsytle choices here
params = {'legend.fontsize': 13,
          'axes.labelsize': 18}
plt.rcParams.update(params)

def histogramOverlay(ax, data, labels, xlabel, ylabel,
                     x_min = 0, x_max = 2200, xbins = 22,
                     normed = True, y_log = False,
                     atlas_x = -1, atlas_y = -1, simulation = False,
                     textlist = [],
                     ps = qu.PlotStyle('dark')):
    
    xbin = np.arange(x_min, x_max, (x_max - x_min) / xbins)
    zorder_start = -1 * len(data) # hack to get axes on top
    colors = ps.colors
    
    for i,vals in enumerate(data):
        ax.hist(vals, bins = xbin, density = normed, 
                alpha = 0.5, label=labels[i], 
                color = colors[i%len(colors)], zorder=zorder_start + i)
    
    ax.set_xlim(x_min,x_max)
    
    if y_log: ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ps.SetStylePlt(ax)

    # TODO: find a way to replace this
    #if atlas_x >= 0 and atlas_y >= 0:
        #ampl.draw_atlas_label(atlas_x, atlas_y, simulation = simulation, fontsize = 18)

    #drawLabels(fig, atlas_x, atlas_y, simulation, textlist) #TODO: fix for fig,ax implementation
    
    ax.set_zorder = len(data)+1 #hack to keep the tick marks up
    legend = ax.legend(facecolor=ps.canv_plt)
    for leg_text in legend.get_texts(): leg_text.set_color(ps.text_plt)
    return

def multiplot_common(ax, xcenter, lines, labels, xlabel, ylabel,
                    x_min = None, x_max = None, 
                    y_min = None, y_max = None,
                    x_log = False, y_log = False,
                    x_ticks = None,
                    linestyles=[], colorgrouping=-1,
                    extra_lines = [],
                    atlas_x=-1, atlas_y=-1, simulation=False,
                    textlist=[],
                    title = '',
                    ps = qu.PlotStyle('dark')):
    '''
    Creates a set of plots, on a common carrier "xcenter".
    Draws the plots on a provided axis.
    '''
    
    if(x_min == None): x_min = np.min(xcenter)
    if(x_max == None): x_max = np.max(xcenter)
        
    if(y_min == None): y_min = np.minimum(0.,np.min(np.column_stack(lines)))
    if(y_max == None): y_max = 1.25 * np.max(np.column_stack(lines))
        
    if(x_ticks != None): ax.xaxis.set_major_locator(plt.MaxNLocator(x_ticks))

    for extra_line in extra_lines:
        ax.plot(extra_line[0], extra_line[1], linestyle='--', color='black')

    colors = ps.colors

    for i, line in enumerate(lines):
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor(i  / colorgrouping))]
        else:
            color = colors[i % len(colors)]
        ax.plot(xcenter, line, label = labels[i], linestyle=linestyle,color=color)

    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ps.SetStylePlt(ax)
    
    #drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

    legend = ax.legend(facecolor=ps.canv_plt)
    for leg_text in legend.get_texts(): leg_text.set_color(ps.text_plt)
    return

def multiplot(ax, xlist, ylist, 
              xlabel='False positive rate', ylabel='True positive rate', 
              x_min = 0, x_max = 1.1, x_log = False,
              y_min = 0, y_max = 1.1, y_log = False, 
              linestyles=[], colorgrouping=-1, 
              extra_lines=[], labels=[], 
              atlas_x=-1, atlas_y=-1, simulation=False,  
              textlist=[], title='', 
              ps = qu.PlotStyle('dark')):
    
    '''
    Creates a set of plots, from series of x and y values (does not use a common carrier).
    Draws the plots on a provided axis.
    '''
    for extra_line in extra_lines:
        ax.plot(extra_line[0], extra_line[1], linestyle='--', color=ps.main_plt)
        
    colors = ps.colors

    for i, (x,y) in enumerate(zip(xlist,ylist)):
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor(i / colorgrouping))]
        else:
            color = colors[i%(len(colors)-1)]
        label = None
        if len(labels) > 0:
            label = labels[i]
        ax.plot(x, y, label = label, linestyle=linestyle, color=color)
        
    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')
        
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ps.SetStylePlt(ax)
    
    legend = ax.legend(facecolor=ps.canv_plt)
    for leg_text in legend.get_texts(): leg_text.set_color(ps.text_plt)
        
    #drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    return
    
def roc_plot(ax, xlist, ylist,
             xlabel='False positive rate',
             ylabel='True positive rate',
             x_min = 0, x_max = 1.1, x_log = False,
             y_min = 0, y_max = 1.1, y_log = False,
             linestyles=[], colorgrouping=-1,
             extra_lines=[[[0, 1], [0, 1]]], labels=[],
             atlas_x=-1, atlas_y=-1, simulation=False,
             textlist=[], title='',
             ps = qu.PlotStyle('dark')):
    '''
    Shortcut for making a ROC curve.
    '''
    multiplot(ax, xlist, ylist,
              xlabel=xlabel,
              ylabel=ylabel,
              x_min=x_min, x_max=x_max, x_log=x_log,
              y_min=y_min, y_max=y_max, y_log=y_log,
              linestyles=linestyles, colorgrouping=colorgrouping,
              extra_lines=extra_lines, labels=labels,
              atlas_x=atlas_x, atlas_y=atlas_y, simulation=simulation,
              textlist=textlist, title=title,
              ps=ps
    )
    return
    
# Plot multiple data series, with x bins as integer array (0,...N-1)
def make_plot(items, figfile = '',
              xlabel = '', ylabel = '',
              x_log = False, y_log = False,
              labels = [], title = '',
              ps = qu.PlotStyle('dark')):
    plt.cla()
    plt.clf()
    
    fig,ax = plt.subplots(1,1)
    colors = ps.colors
    #fig.patch.set_facecolor('white')
    for i, item in enumerate(items):
        label = None
        if len(labels) >= i:
            label = labels[i]
        color = colors[i%len(colors)]
        ax.plot(item, label=label,color=color)
        
    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()
    if figfile != '':
        plt.savefig(figfile,transparent=True, facecolor=ps.canv_plt)
    plt.show()
    
def drawLabels(fig, atlas_x=-1, atlas_y=-1, simulation=False,
               textlist=[]):
    
    # TODO: find a non-ampl way to do this
#     if atlas_x >= 0 and atlas_y >= 0:
#         ampl.draw_atlas_label(atlas_x, atlas_y, simulation=simulation, fontsize=18)

    for textdict in textlist:
        fig.axes[0].text(
            textdict['x'], textdict['y'], textdict['text'], 
            transform=fig.axes[0].transAxes, fontsize=18)


display_digits = 2


class rocVar:
    def __init__(self,
                 name,  # name of variable as it appears in the root file
                 bins,  # endpoints of bins as a list
                 df,   # dataframe to construct subsets from
                 latex='',  # optional latex to display variable name with
                 vlist=None,  # optional list to append class instance to
                 ):
        self.name = name
        self.bins = bins

        if(latex == ''):
            self.latex = name
        else:
            self.latex = latex

        self.selections = []
        self.labels = []
        for i, point in enumerate(self.bins):
            if(i == 0):
                self.selections.append(df[name] < point)
                self.labels.append(
                    self.latex+'<'+str(round(point, display_digits)))
            else:
                self.selections.append(
                    (df[name] > self.bins[i-1]) & (df[name] < self.bins[i]))
                self.labels.append(str(round(
                    self.bins[i-1], display_digits))+'<'+self.latex+'<'+str(round(point, display_digits)))
                if(i == len(bins)-1):
                    self.selections.append(df[name] > point)
                    self.labels.append(
                        self.latex+'>'+str(round(point, display_digits)))

        if(vlist != None):
            vlist.append(self)


def rocScan(varlist, scan_targets, labels, ylabels, data, plotpath='',
            x_min=0., x_max=1.0, y_min=0.0, y_max=1.0, x_log = False, y_log = False, rejection = False,
            x_label = 'False positive rate', y_label = 'True positive rate',
            linestyles=[], colorgrouping=-1,
            extra_lines=[],
            atlas_x=-1, atlas_y=-1, simulation=False,
            textlist=[]):
    '''
    Creates a set of ROC curve plots by scanning over the specified variables.
    One set is created for each target (neural net score dataset).
    
    varlist: a list of rocVar instances to scan over
    scan_targets: a list of neural net score datasets to use
    labels: a list of target names (strings); must be the same length as scan_targets
    '''

    rocs = buildRocs(varlist, scan_targets, labels, ylabels, data)

    for target_label in labels:
        for v in varlist:
            # prepare matplotlib figure
            plt.cla()
            plt.clf()
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            plt.plot([0, 1], [0, 1], 'k--')

            for label in v.labels:
                # first generate ROC curve
                x = rocs[target_label+label]['x']
                y = rocs[target_label+label]['y']
                var_auc = auc(x, y)
                if not rejection:
                    plt.plot(x, y, label=label+' (area = {:.3f})'.format(var_auc))
                else:
                    plt.plot(y, 1. / x, label=label +
                             ' (area = {:.3f})'.format(var_auc))

            # plt.title('ROC Scan of '+target_label+' over '+v.latex)
            if x_log:
                plt.xscale('log')
            if y_log:
                plt.yscale('log')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            plt.x_label(x_label)
            plt.y_label(y_label)
            
            #ampl.set_xlabel(x_label)
            #ampl.set_ylabel(y_label)
            plt.legend()

            drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

            if plotpath != '':
                plt.savefig(plotpath+'roc_scan_' +
                            target_label+'_'+v.name+'.pdf')
            plt.show()

def buildRocs(varlist, scan_targets, labels, ylabels, data):
    rocs = {}
    for target, target_label in zip(scan_targets, labels):
        for v in varlist:
            for binning, label in zip(v.selections, v.labels):
                # first generate ROC curve
                x, y, t = roc_curve(
                    ylabels[data.test & binning][:, 1],
                    target[data.test & binning],
                    drop_intermediate=False,
                )

                rocs[target_label + label] = {'x': x, 'y': y}

    return rocs
