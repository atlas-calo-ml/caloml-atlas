import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import plot_util as pu
from util import ml_util as mu
from util import qol_util  as qu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable some of the tensorflow info printouts, only display errors
import tensorflow as tf
from util.keras.layers import ImageScaleBlock

# Create plots of accuracy and loss.
def MetricPlot(model_history, 
               acc_range=(0.5,1.), loss_range=(0.,0.7), 
               acc_log=False, loss_log=False, 
               plotpath='/', 
               model_keys=[], 
               plotstyle=qu.PlotStyle('dark')):
    
    if(model_keys == []): model_keys = list(model_history.keys())
    for model_key in model_keys:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
    
        keys = ['acc','val_acc']
        lines = [model_history[model_key][key] for key in keys]
        epochs = np.arange(len(lines[0])) + 1
        pu.multiplot_common(
            ax[0], 
            epochs,
            lines, 
            keys, 
            y_min = acc_range[0], 
            y_max = acc_range[1],
            y_log = acc_log,
            xlabel = 'Epoch',
            ylabel = 'Accuracy',
            title='Model accuracy for {}'.format(model_key),
            ps=plotstyle
        )
    
        keys = ['loss','val_loss']
        lines = [model_history[model_key][key] for key in keys]
        pu.multiplot_common(
            ax[1], 
            epochs,
            lines, 
            keys, 
            y_min = loss_range[0], 
            y_max = loss_range[1],
            y_log = loss_log,
            xlabel = 'Epoch', 
            ylabel = 'Loss', 
            title='Model loss for {}'.format(model_key), 
            ps=plotstyle
        )
    
        # add grids
        for axis in ax.flatten():
            axis.grid(True,color=plotstyle.grid_plt)

        qu.SaveSubplots(fig, ax, ['accuracy_{}'.format(model_key), 'loss_{}'.format(model_key)], savedir=plotpath, ps=plotstyle)
        plt.show()
    return

# Create ROC curves. Note that some of the arguments are dictionaries, that get modified.
def RocCurves(model_scores, 
              data_labels, 
              roc_fpr, roc_tpr, 
              roc_thresh, roc_auc, 
              indices=[], 
              plotpath = '/', plotname = 'ROC', 
              model_keys = [], 
              drawPlots=True, figsize=(15,5), 
              plotstyle=qu.PlotStyle('dark')):
    
    if(model_keys == []): model_keys = list(model_scores.keys())
    if(len(indices) != len(data_labels)):  indices = np.full(len(data_labels), True, dtype=np.dtype('bool'))
        
    for model_key in model_keys:
        roc_fpr[model_key], roc_tpr[model_key], roc_thresh[model_key] = roc_curve(
            data_labels[indices],
            model_scores[model_key][indices],
            drop_intermediate=False,
        )
        roc_auc[model_key] = auc(roc_fpr[model_key], roc_tpr[model_key])
        print('Area under curve for {}: {}'.format(model_key, roc_auc[model_key]))
        
    if(not drawPlots): return
    
    # TODO: Sort model_keys by AUC
        
    # Make a plot of the ROC curves
    fig, ax = plt.subplots(1,2,figsize=figsize)
    xlist = [roc_fpr[x] for x in model_keys]
    ylist = [roc_tpr[x] for x in model_keys]
    labels = ['{} (area = {:.3f})'.format(x, roc_auc[x]) for x in model_keys]
    title = 'ROC curve: classification of $\pi^+$ vs. $\pi^0$'

    pu.roc_plot(ax[0], 
                xlist=xlist, 
                ylist=ylist,
                labels=labels,
                title=title,
                ps=plotstyle
                )
    
    title = 'ROC curve (zoomed in at top left)'
    pu.roc_plot(ax[1], 
                xlist=xlist, 
                ylist=ylist,
                x_min=0. , x_max=0.25,
                y_min=0.6, y_max=1.,
                labels=labels,
                title=title,
                ps=plotstyle
                )
    qu.SaveSubplots(fig, ax, [plotname, plotname + '_zoom'], savedir=plotpath, ps=plotstyle)
    plt.show()
    return


# -- Kinematic Plots below --
def ImagePlot(pcells, cluster, log=True, dynamic_range=False, layers=[], cell_shapes={}, scaled_shape = [], latex_mpl = {}, plotpath = '', filename = '', plotstyle=qu.PlotStyle('dark')):
    # Set some default values.
    if(layers == []): layers = list(mu.cell_meta.keys())
    if(cell_shapes == {}): cell_shapes = {key: (val['len_eta'],val['len_phi']) for key,val in mu.cell_meta.items()}
    
    if(latex_mpl == {}):
        latex_mpl = {
            'p0': '$\pi^{0}$',
            'pp': '$\pi^{+}$'
        }
    
    scaling = False
    if(scaled_shape != []): scaling = True
    
    fig, ax = plt.subplots(len(pcells.keys()),len(layers),figsize=(60,20))
    fig.patch.set_facecolor(plotstyle.canv_plt)
    i = 0
    for ptype, pcell in pcells.items():
        for layer in layers:
            axis = ax.flatten()[i]
            
            # default behaviour: plot a single cluster
            if(cluster >= 0): image = pcell[layer][cluster].reshape(cell_shapes[layer])
            
            # if cluster index is negative, provide an average image
            else: image = np.mean(pcell[layer],axis=0).reshape(cell_shapes[layer])
            
            if(dynamic_range):
                vmin, vmax = np.min(image), np.max(image)
                vmax = np.maximum(np.abs(vmin),np.abs(vmax))
                vmin = -vmax

                if(vmax == 0. and vmin == 0.):
                    vmax = 0.1
                    vmin = -vmax
                    
            else: vmin, vmax = (-1.,1.)

            norm = TwoSlopeNorm(vmin=vmin,vcenter=0.,vmax=vmax)
            if(log): norm = SymLogNorm(linthresh = 0.001, linscale=0.001, vmin=vmin, vmax=vmax, base=10.)
            cmap = plt.get_cmap('BrBG')
            
            image = pcell[layer][cluster].reshape(cell_shapes[layer])
            if(scaling): 
                # Use our ImageScaleBlock. It requires a 4d tensor, format is [batch,eta,phi,channel].
                image = np.expand_dims(image, axis=(0,-1))
                image = np.squeeze(ImageScaleBlock(new_shape=tuple(scaled_shape), normalization=True)([image]).numpy())
  
            im = axis.imshow(
                image, 
                extent=[-0.2, 0.2, -0.2, 0.2],
                cmap=cmap, 
                origin='lower', 
                interpolation='nearest',
                norm=norm
            )
            #axis.colorbar()
            axis.set_title('{a} in {b}'.format(a=latex_mpl[ptype],b=layer))
            axis.set_xlabel("$\Delta\phi$")
            axis.set_ylabel("$\Delta\eta$")

            plotstyle.SetStylePlt(axis)
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            cb = fig.colorbar(im, cax=cax, orientation='vertical')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=plotstyle.text_plt)
            i += 1
            
    # show the plots
    if(filename != ''): plt.savefig('{}/{}'.format(plotpath,filename),transparent=True,facecolor=plotstyle.canv_plt)
    plt.show()
    return