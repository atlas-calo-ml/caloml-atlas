import os
import ROOT as rt
import numpy as np
import matplotlib.pyplot as plt

from util import plot_util as pu
from util import qol_util  as qu

def Median(values, counts, default = 1):
    nvar = len(values)
    assert(len(counts) == nvar)
    if(nvar == 0): return default
    value_list = [[values[i] for j in range(int(counts[i]))] for i in range(nvar)]
    value_array = [item for sublist in value_list for item in sublist]
    value_array = np.array(value_array)
    if(len(value_array) == 0): return default
    med = np.median(value_array)
    return med

# a simple version of Max's cool 2D plots
def EnergyPlot2D(e1, e2, title='title;x;y', nbins = [100,35], x_range = [0.,2000.], y_range = [0.3, 1.7], offset=False, mode = 'median'):
    hist = rt.TH2F(qu.RN(), title, nbins[0], x_range[0], x_range[1], nbins[1], y_range[0], y_range[1])
    x_vals = e2
    y_vals = e1/e2
    if(offset): x_vals = x_vals + 1.
    
    for i in range(len(e2)):
        hist.Fill(x_vals[i],y_vals[i])
        
    # now we want to make a curve representing the medians or means in y (median by default, despite variable names!)
    bin_centers_y = np.array([hist.GetYaxis().GetBinCenter(i+1) for i in range(nbins[1])])
    mean_vals = np.zeros(nbins[0])
    for i in range(nbins[0]):
        weights = np.array([hist.GetBinContent(i+1, j+1) for j in range(nbins[1])])
        if(mode == 'median'):
            mean_vals[i] = Median(bin_centers_y,np.array(weights,dtype=np.dtype('i8')))
        else:
            if(np.sum(weights) != 0.):
                weights = weights / np.sum(weights)
                y_vals = np.array([hist.GetYaxis().GetBinCenter(j+1) for j in range(nbins[1])])
                y_vals = np.multiply(y_vals,weights)
                mean_vals[i] = np.sum(y_vals)
            else: mean_vals[i] = 1.
        
    x_vals = np.array([hist.GetXaxis().GetBinCenter(i+1) for i in range(nbins[0])])
    curve = rt.TGraph(nbins[0],x_vals, mean_vals)
    curve.SetLineColor(rt.kRed)
    curve.SetLineWidth(2)
    return curve, hist

# For plotting the inter-quantile range for ratio of predicted energy to true energy, versus true energy.
# (interquantile, *not* interquartile)
def IqrPlot(e1, e2, title='title;x;y', nbins = 100, x_range = [0.,2000.], offset=False, quantiles = (16, 84), normalize=True):
    
    # To compute the true IQR's, we will use lists of energy ratios for each truth energy bin,
    # versus using the 2D histogram from EnergyPlot2D -- we don't want to bin in y, because that
    # will affect the resolution of the IQR that we calculate.
    n = len(e1)
    if(offset): e2 = e2 + 1.
    x_bins = np.linspace(*x_range,nbins)
    y_lists = [[] for i in range(nbins)]
    
    # get the binned x_vals
    x_vals = np.digitize(e2,x_bins) - 1 # subtracting one for the expected binning behaviour, e.g. with x_bins[0]=0 & x_bins[1]=1, x=.5 should go in bin[0]
    
    ratios = e1/e2
    for i in range(n):
        y_lists[int(x_vals[i])].append(ratios[i])
    y_lists = [np.array(x) for x in y_lists]
    
    # Now calculate the IQR for each x bin.
    iqr =  np.array([np.percentile(x,q=quantiles[1]) for x in y_lists])
    iqr -= np.array([np.percentile(x,q=quantiles[0]) for x in y_lists])
        
    # Also fetch the medians, which we might need if normalizing by median.
    median = [np.median(x) for x in y_lists]
        
    # Now make the plot.     
    # Scipy.stats.iqr gives nan for empty input,
    # so we need to avoid these and leave the 
    # corresponding bins empty.
    hist = rt.TH1F(qu.RN(),title,nbins,*x_range)
    for i in range(nbins):
        val = iqr[i]
        if(np.isnan(val)): continue
        if(normalize): val = val / median[i]
        b = i+1
        hist.SetBinContent(b,val)        
    return hist

# For plotting the median ratio of predicted energy to true energy, versus true energy
# (Complements the EnergyPlot2D() output)
def MedianPlot(e1, e2, title='title;x;y', nbins = 100, x_range = [0.,2000.], offset=False):
    
    # To compute the true medians, we will use lists of energy ratios for each truth energy bin,
    # versus using the 2D histogram from EnergyPlot2D -- we don't want to bin in y, because that
    # will affect the resolution of the median that we calculate.
    n = len(e1)
    if(offset): e2 = e2 + 1.
    x_bins = np.linspace(*x_range,nbins)
    y_lists = [[] for i in range(nbins)]
    
    # get the binned x_vals
    x_vals = np.digitize(e2,x_bins) - 1 # subtracting one for the expected binning behaviour, e.g. with x_bins[0]=0 & x_bins[1]=1, x=.5 should go in bin[0]
    
    ratios = e1/e2
    for i in range(n):
        y_lists[int(x_vals[i])].append(ratios[i])
    y_lists = [np.array(x) for x in y_lists]
    
    # Now calculate the IQR for each x bin.
    median = [np.median(x) for x in y_lists]
        
    # Now make the plot.     
    # Scipy.stats.iqr gives nan for empty input,
    # so we need to avoid these and leave the 
    # corresponding bins empty.
    hist = rt.TH1F(qu.RN(),title,nbins,*x_range)
    for i in range(nbins):
        val = median[i]
        if(np.isnan(val)): continue
        b = i+1
        hist.SetBinContent(b,val)
    return hist

# Big plotting function. #TODO: Should we break this up into parts?
def EnergySummary(train_dfs, valid_dfs, data_dfs, energy_name, model_name, plotpath, extensions=['png'], plot_size=750, full=True, ps=qu.PlotStyle('dark'), **kwargs):
    
    ps.SetStyle()
    
    max_energy = 2000. # GeV
    max_energy_2d = max_energy
    bin_energy = 300
    ratio_range_2d = [0.3, 1.7]
    bins_2d = [200,70]
    
    if('max_energy' in kwargs.keys()): 
        max_energy = kwargs['max_energy']
        max_energy_2d = max_energy
    if('max_energy_2d' in kwargs.keys()): max_energy = kwargs['max_energy_2d']
    if('bin_energy' in kwargs.keys()): bin_energy = kwargs['bin_energy']
    if('ratio_range_2d' in kwargs.keys()): ratio_range_2d = kwargs['ratio_range_2d']
    if('bins_2d' in kwargs.keys()): bins_2d = kwargs['bins_2d']
    
    # Dictionaries to keep track of all our histogram objects.
    # Each entry will be a dictionary of hists.
    # Outer key is data type (charged pion, neutral pion),
    # inner key is data set (train, valid, all).
    
    clusterE = {}                # reco energy
    clusterE_calib = {}          # cluster_ENG_CALIB_TOT (true energy, as far as we're concerned)
    clusterE_pred = {}           # predicted energy
    #clusterE_true = {}           # "truth" energy from the parton level (I think), not what we're after
    
    clusterE_ratio1 = {}         # ratio1: E_pred / ENG_CALIB_TOT
    clusterE_ratio2 = {}         # ratio2: E_reco / ENG_CALIB_TOT
    clusterE_ratio2D = {}        # ratio1 vs. ENG_CALIB_TOT
    clusterE_ratio2D_zoomed = {} # ratio1 vs. ENG_CALIB_TOT (Zoomed on left)
    
    ratio1_iqr = {}              # IQR, from ratio1
    ratio2_iqr = {}              # IQR, from ratio2
    ratio1_iqr_zoomed = {}              # IQR, from ratio1
    ratio2_iqr_zoomed = {}              # IQR, from ratio2
    
    # histogram stacks
    energy_stacks = {}
    iqr_stacks = {}
    iqr_stacks_zoomed = {}

    # keep track of mean/median curves from the 2D plots (one set for each).
    mean_curves = {}
    mean_curves_zoomed = {}

    # keep track of our canvases, legends and histogram stacks
    canvs = {}
    legends = {}

    key_conversions = {
        'pp':'#pi^{#pm}',
        'p0':'#pi^{0}',
    }

    dsets = {
        'train': train_dfs,
        'valid': valid_dfs,
        'all data': data_dfs
    }

    for key in train_dfs.keys(): # assuming all DataFrame dicts have the same keys
        
        # Initialize the inner dictionaries.
        clusterE[key] = {}
        clusterE_calib[key] = {}
        clusterE_pred[key] = {}    
        #clusterE_true[key] = {}
        
        clusterE_ratio1[key] = {}
        clusterE_ratio2[key] = {}
        clusterE_ratio2D[key] = {}
        clusterE_ratio2D_zoomed[key] = {}
    
        ratio1_iqr[key] = {}
        ratio2_iqr[key] = {}
        ratio1_iqr_zoomed[key] = {}
        ratio2_iqr_zoomed[key] = {}
        
        energy_stacks[key] = {}
        iqr_stacks[key] = {}
        iqr_stacks_zoomed[key] = {}

        mean_curves[key] = {}
        mean_curves_zoomed[key] = {}
    
        canvs[key] = {}
        legends[key] = {}
        iqr_stacks[key] = {}
    
        for dkey, frame in dsets.items():
            key2 = '(' + key_conversions[key] + ', ' + dkey + ')'
            clusterE[key][dkey] = rt.TH1F(qu.RN(), 'E_{reco} ' + key2 +'; E_{reco} [GeV];Count' , bin_energy,0.,max_energy)
            clusterE_calib[key][dkey] = rt.TH1F(qu.RN(), 'E_{calib}^{tot} ' + key2 + ';E_{calib}^{tot} [GeV];Count', bin_energy,0.,max_energy)
            clusterE_pred[key][dkey] = rt.TH1F(qu.RN(), 'E_{pred} ' + key2 + ';E_{pred} [GeV];Count', bin_energy,0.,max_energy)
            #clusterE_true[key][dkey] = rt.TH1F(qu.RN(), 'E_{true} ' + key2 + ';E_{true} [GeV];Count', bin_energy,0.,max_energy)
            clusterE_ratio1[key][dkey] = rt.TH1F(qu.RN(), 'E_{pred} / E_{calib}^{tot} ' + key2 + ';E_{pred}/E_{calib}^{tot};Count', 250,0.,10.)
            clusterE_ratio2[key][dkey] = rt.TH1F(qu.RN(), 'E / E_{calib}^{tot} ' + key2 + ';E_{reco}/E_{calib}^{tot]};Count', 250,0.,10.)

            qu.SetColor(clusterE[key][dkey], ps.main, alpha = 0.4)
            qu.SetColor(clusterE_calib[key][dkey], rt.kPink + 9, alpha = 0.4)
            qu.SetColor(clusterE_pred[key][dkey], ps.curve, alpha = 0.4)        
#             qu.SetColor(clusterE_true[key][dkey], rt.kRed, alpha = 0.4)
            qu.SetColor(clusterE_ratio1[key][dkey], ps.main, alpha = 0.4)
            qu.SetColor(clusterE_ratio2[key][dkey], ps.curve, alpha = 0.4)

            meas   = frame[key]['clusterE'].to_numpy()
            calib  = frame[key]['cluster_ENG_CALIB_TOT'].to_numpy()
            pred   = frame[key][energy_name].to_numpy()
            #true   = frame[key]['truthE'].to_numpy()
            ratio1 = pred / calib
            ratio2 = meas / calib
    
            for i in range(len(meas)):
                clusterE[key][dkey].Fill(meas[i])
                clusterE_calib[key][dkey].Fill(calib[i])
                clusterE_pred[key][dkey].Fill(pred[i])            
                #clusterE_true[key][dkey].Fill(true[i])
                clusterE_ratio1[key][dkey].Fill(ratio1[i])
                clusterE_ratio2[key][dkey].Fill(ratio2[i])
            
            # Fill the histogram stack for the energy ratios.
            energy_stacks[key][dkey] = rt.THStack(qu.RN(), clusterE_ratio1[key][dkey].GetTitle())
            energy_stacks[key][dkey].Add(clusterE_ratio1[key][dkey])
            energy_stacks[key][dkey].Add(clusterE_ratio2[key][dkey])

            # Make the 2D energy ratio plots.
            title = 'E_{pred}/E_{calib}^{tot} vs. E_{calib}^{tot} ' + key2 + ';E_{calib}^{tot} [GeV];E_{pred}/E_{calib}^{tot};Count'
            x_range = [0.,max_energy_2d]
            nbins = bins_2d
            mean_curves[key][dkey], clusterE_ratio2D[key][dkey] = EnergyPlot2D(pred, calib, nbins = nbins, x_range = x_range, y_range = ratio_range_2d, title=title, offset=True)

            title = 'E_{pred}/E_{calib}^{tot} vs. E_{calib}^{tot} ' + key2 + ';(E_{calib}^{tot} + 1) [GeV];E_{pred}/E_{calib}^{tot};Count'
            x_range = [1.,1. + 0.01 * max_energy_2d]
            nbins = bins_2d
            nbins[0] = nbins[0] - 1
            mean_curves_zoomed[key][dkey], clusterE_ratio2D_zoomed[key][dkey] = EnergyPlot2D(pred, calib, nbins = nbins, x_range = x_range, y_range = ratio_range_2d, title=title, offset=False)        
        
            # Make the energy ratio IQR plots.
            title = 'IQR(E_{x}/E_{calib}^{tot}) ' + key2 + ';E_{calib}^{tot} [GeV];IQR'
            x_range = [0.,max_energy_2d]
            nbins = int(bins_2d[0]/2)
            ratio1_iqr[key][dkey] = IqrPlot(pred, calib, title=title, nbins=nbins, x_range=x_range)
            ratio1_iqr[key][dkey].SetLineColor(ps.main)
            ratio2_iqr[key][dkey] = IqrPlot(meas, calib, title=title, nbins=nbins, x_range=x_range)
            ratio2_iqr[key][dkey].SetLineColor(ps.curve)
            
            title = 'IQR(E_{x}/E_{calib}^{tot}) ' + key2 + ';(E_{calib}^{tot} + 1) [GeV];IQR'
            x_range = [1.,1. + 0.01 * max_energy_2d]
            nbins = int((bins_2d[0] - 1)/2)
            ratio1_iqr_zoomed[key][dkey] = IqrPlot(pred, calib, title=title, nbins=nbins, x_range=x_range, offset=True)
            ratio1_iqr_zoomed[key][dkey].SetLineColor(ps.main)
            ratio2_iqr_zoomed[key][dkey] = IqrPlot(meas, calib, title=title, nbins=nbins, x_range=x_range, offset=True)
            ratio2_iqr_zoomed[key][dkey].SetLineColor(ps.curve)
            
            # Fill the histogram stack for the IQR plots.
            iqr_stacks[key][dkey] = rt.THStack(qu.RN(),title)
            iqr_stacks[key][dkey].Add(ratio1_iqr[key][dkey])
            iqr_stacks[key][dkey].Add(ratio2_iqr[key][dkey])
            
            iqr_stacks_zoomed[key][dkey] = rt.THStack(qu.RN(),title)
            iqr_stacks_zoomed[key][dkey].Add(ratio1_iqr_zoomed[key][dkey])
            iqr_stacks_zoomed[key][dkey].Add(ratio2_iqr_zoomed[key][dkey])
            
        # Prepare the list of plots we'll show (we might exclude some).
        plots = [clusterE, clusterE_calib, clusterE_pred, energy_stacks, clusterE_ratio2D, clusterE_ratio2D_zoomed, iqr_stacks, iqr_stacks_zoomed]
        if(not full): plots = [energy_stacks, clusterE_ratio2D, clusterE_ratio2D_zoomed, iqr_stacks, iqr_stacks_zoomed]
        dkeys = list(dsets.keys())
        
        # Make legend for the overlapping plots (1D energy ratios, and IQR plots)
        legends[key] = rt.TLegend(0.7,0.7,0.85,0.85)
        legends[key].SetBorderSize(0)
        legends[key].AddEntry(clusterE_ratio1[key][dkeys[0]],'x = pred','f')
        legends[key].AddEntry(clusterE_ratio2[key][dkeys[0]],'x = reco','f')
    
        nx = len(dkeys)
        ny = len(plots)
        canvs[key] = rt.TCanvas(qu.RN(),'c_'+str(key),nx * plot_size,ny * plot_size)
        canvs[key].Divide(nx,ny)
    
        for i, plot in enumerate(plots):
            x = nx * i + 1
            if(plot == energy_stacks or plot == iqr_stacks or plot == iqr_stacks_zoomed):
                for j, dkey in enumerate(dkeys):
                    canvs[key].cd(x + j)
                    
                    draw_option = 'NOSTACK HIST'
                    if(plot != energy_stacks): draw_option = 'NOSTACK C'
                    plot[key][dkey].Draw(draw_option)
                    
                    rt.gPad.SetGrid()
                    rt.gPad.SetLogy()
                    plot[key][dkey].GetHistogram().GetXaxis().SetTitle('E_{x}/E_{calib}^{tot}')

                    if(plot == energy_stacks):
                        plot[key][dkey].GetHistogram().GetYaxis().SetTitle(clusterE_ratio1[key][dkey].GetYaxis().GetTitle())
                        
#                         if(strat == 'jet'):
#                             plot[key][dkey].SetMinimum(5.0e-1)
#                             plot[key][dkey].SetMaximum(1.0e3)
                    
                        plot[key][dkey].SetMinimum(5.0e-1)
                        plot[key][dkey].SetMaximum(2.0e5)   
                    
                    else:
                        plot[key][dkey].GetHistogram().GetYaxis().SetTitle(ratio1_iqr[key][dkey].GetYaxis().GetTitle())
                        plot[key][dkey].SetMinimum(1.0e-2)
                        plot[key][dkey].SetMaximum(1.)
                        
                    if(plot == iqr_stacks_zoomed):
                        rt.gPad.SetLogx()
                        rt.gPad.SetBottomMargin(0.15)
                        plot[key][dkey].GetXaxis().SetTitleOffset(1.5)
                        
                    if(plot == iqr_stacks or plot == iqr_stacks_zoomed):
                        plot[key][dkey].SetMinimum(1.0e-3)
                        
                    legends[key].SetTextColor(ps.text)
                    legends[key].Draw()

            elif(plot == clusterE_ratio2D or plot == clusterE_ratio2D_zoomed):
                for j, dkey in enumerate(dkeys):
                    canvs[key].cd(x + j)
                    plot[key][dkey].Draw('COLZ')
                    if(plot == clusterE_ratio2D): mean_curves[key][dkey].Draw('SAME')
                    else: 
                        mean_curves_zoomed[key][dkey].Draw('SAME')
                        rt.gPad.SetLogx()
                        rt.gPad.SetBottomMargin(0.15)
                        plot[key][dkey].GetXaxis().SetTitleOffset(1.5)
                    
                    rt.gPad.SetLogz()
                    rt.gPad.SetRightMargin(0.2)
                    plot[key][dkey].GetXaxis().SetMaxDigits(4)
                  
            else:
                for j, dkey in enumerate(dkeys):
                    canvs[key].cd(x + j)
                    plot[key][dkey].Draw('HIST')
                    plot[key][dkey].SetMinimum(5.0e-1)
                    rt.gPad.SetLogy()
    
        # Draw the canvas
        canvs[key].Draw()
    
        # Save the canvas as a PDF & PNG image.
        image_name = '_'.join([model_name,key,'plots'])
        for ext in extensions: canvs[key].SaveAs(plotpath + image_name + '.' + ext)
            
    results = {}
    results['canv'] = canvs
    results['plots'] = plots
    results['curves'] = [mean_curves, mean_curves_zoomed]
    results['legend'] = legends
    return results