import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import plot_util as pu
from util import qol_util  as qu

# Create plots of accuracy and loss.
def MetricPlot(model_history, acc_range=(0.5,1.), loss_range=(0.,0.7), acc_log=False, loss_log=False, plotpath='/', model_keys=[], plotstyle=qu.PlotStyle('dark')):
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
def RocCurves(model_scores, data_labels, roc_fpr, roc_tpr, roc_thresh, roc_auc, indices=[], plotpath = '/', plotname = 'ROC', model_keys = [], drawPlots=True, figsize=(15,5), plotstyle=qu.PlotStyle('dark')):
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
def ImagePlot(pcells, cluster, layers=[], cell_shapes=[], latex_mpl = {}, plotpath = '', filename = 'plots_pi0_plus_minus.png', plotstyle=qu.PlotStyle('dark')):
    # Set some default values.
    if(layers == []):
        layers = ['EMB1','EMB2','EMB3','TileBar0','TileBar1','TileBar2']
    
    if(cell_shapes == []):
        len_phi = [4, 16, 16, 4, 4, 4]
        len_eta = [128, 16, 8, 4, 4, 2]
        cell_shapes = {layers[i]:(len_eta[i],len_phi[i]) for i in range(len(layers))}
    
    if(latex_mpl == {}):
        latex_mpl = {
            'pi0': '$\pi^{0}$',
            'piplus': '$\pi^{+}$'
        }
    
    fig, ax = plt.subplots(2,6,figsize=(60,20))
    fig.patch.set_facecolor(plotstyle.canv_plt)
    i = 0
    for ptype, pcell in pcells.items():
        for layer in layers:
            axis = ax.flatten()[i]
            image = pcell[layer][cluster].reshape(cell_shapes[layer])
            vmin, vmax = np.min(image), np.max(image)
            vmax = np.maximum(np.abs(vmin),np.abs(vmax))
            vmin = -vmax

            if(vmax == 0. and vmin == 0.):
                vmax = 0.1
                vmin = -vmax

            norm = TwoSlopeNorm(vmin=vmin,vcenter=0.,vmax=vmax)
            cmap = plt.get_cmap('BrBG')
            im = axis.imshow(
                pcell[layer][cluster].reshape(cell_shapes[layer]), 
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
    plt.savefig('{}/{}'.format(plotpath,filename),transparent=True,facecolor=plotstyle.canv_plt)
    plt.show()
    return