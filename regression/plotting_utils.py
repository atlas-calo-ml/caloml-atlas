import sys, os
import ROOT as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_prefix = os.getcwd() + '/../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
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
        
    # now we want to make a curve representing the medians/means in y
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


# Big plotting function. #TODO: Should we break this up into parts?
def EnergySummary(train_dfs, valid_dfs, data_dfs, energy_name, model_name, plotpath, extensions=['png'], plot_size=750, strat='pion', full=True, ps=qu.PlotStyle('dark')):
    
    ps.SetStyle()
    
    # some ranges for the plots
    if(strat == 'pion' or strat == 'pion_reweighted'):
        max_energy = 2000. # GeV
        max_energy_2d = max_energy
        bin_energy = 300
        ratio_range_2d = [0.3, 1.7]
        bins_2d = [200,70]
        offset_2d = False
    
    else:
        max_energy = 100
        max_energy_2d = 10.
        bin_energy = 20
        ratio_range_2d = [0., 5.]
        bins_2d = [50,125]
        offset_2d = True
    
    
    
    # keep track of all our histograms
    clusterE = {}
    clusterE_calib = {}
    clusterE_pred = {}
    clusterE_true = {}
    clusterE_ratio1 = {} # ratio of predicted cluster E to calibrated cluster E
    clusterE_ratio2 = {} # ratio of reco cluster E to calibrated cluster E
    clusterE_ratio2D = {} # 2D plot, ratio1 versus calibrated cluster E
    clusterE_ratio2D_zoomed = {} # 2D plot, ratio1 versus calibrated cluster E, zoomed

    # keep track of mean/median curves on 2D plots
    mean_curves = {}
    mean_curves_zoomed = {}

    # keep track of our canvases, legends and histogram stacks
    canvs = {}
    legends = {}
    stacks = {}

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
        clusterE[key] = {}
        clusterE_calib[key] = {}
        clusterE_pred[key] = {}    
        clusterE_true[key] = {}
        clusterE_ratio1[key] = {}
        clusterE_ratio2[key] = {}
        clusterE_ratio2D[key] = {}
        clusterE_ratio2D_zoomed[key] = {}
    
        mean_curves[key] = {}
        mean_curves_zoomed[key] = {}
    
        canvs[key] = {}
        legends[key] = {}
        stacks[key] = {}
    
        for dkey, frame in dsets.items():
            key2 = '(' + key_conversions[key] + ', ' + dkey + ')'
            clusterE[key][dkey] = rt.TH1F(qu.RN(), 'E_{reco} ' + key2 +'; E_{reco} [GeV];Count' , bin_energy,0.,max_energy)
            clusterE_calib[key][dkey] = rt.TH1F(qu.RN(), 'E_{calib}^{tot} ' + key2 + ';E_{calib}^{tot} [GeV];Count', bin_energy,0.,max_energy)
            clusterE_pred[key][dkey] = rt.TH1F(qu.RN(), 'E_{pred} ' + key2 + ';E_{pred} [GeV];Count', bin_energy,0.,max_energy)
            clusterE_true[key][dkey] = rt.TH1F(qu.RN(), 'E_{true} ' + key2 + ';E_{true} [GeV];Count', bin_energy,0.,max_energy)
            clusterE_ratio1[key][dkey] = rt.TH1F(qu.RN(), 'E_{pred} / E_{calib}^{tot} ' + key2 + ';E_{pred}/E_{calib}^{tot};Count', 250,0.,10.)
            clusterE_ratio2[key][dkey] = rt.TH1F(qu.RN(), 'E / E_{calib}^{tot} ' + key2 + ';E_{reco}/E_{calib}^{tot]};Count', 250,0.,10.)

            qu.SetColor(clusterE[key][dkey], rt.kViolet, alpha = 0.4)
            qu.SetColor(clusterE_calib[key][dkey], rt.kPink + 9, alpha = 0.4)
            qu.SetColor(clusterE_pred[key][dkey], rt.kGreen, alpha = 0.4)        
            qu.SetColor(clusterE_true[key][dkey], rt.kRed, alpha = 0.4)
            qu.SetColor(clusterE_ratio1[key][dkey], rt.kViolet, alpha = 0.4)
            qu.SetColor(clusterE_ratio2[key][dkey], rt.kGreen, alpha = 0.4)

            meas   = frame[key]['clusterE'].to_numpy()
            calib  = frame[key]['cluster_ENG_CALIB_TOT'].to_numpy()
            pred   = frame[key][energy_name].to_numpy()
            true   = frame[key]['truthE'].to_numpy()
            ratio1 = pred / calib
            ratio2 = meas / calib
    
            for i in range(len(meas)):
                clusterE[key][dkey].Fill(meas[i])
                clusterE_calib[key][dkey].Fill(calib[i])
                clusterE_pred[key][dkey].Fill(pred[i])            
                clusterE_true[key][dkey].Fill(true[i])
                clusterE_ratio1[key][dkey].Fill(ratio1[i])
                clusterE_ratio2[key][dkey].Fill(ratio2[i])
            
            # fill the histogram stacks
            stacks[key][dkey] = rt.THStack(qu.RN(), clusterE_ratio1[key][dkey].GetTitle())
            stacks[key][dkey].Add(clusterE_ratio1[key][dkey])
            stacks[key][dkey].Add(clusterE_ratio2[key][dkey])

            # 2D plots
            title = 'E_{pred}/E_{calib}^{tot} vs. E_{calib}^{tot} ' + key2 + ';E_{calib}^{tot} [GeV];E_{pred}/E_{calib}^{tot};Count'
            x_range = [0.,max_energy_2d]
            nbins = bins_2d
            mean_curves[key][dkey], clusterE_ratio2D[key][dkey] = EnergyPlot2D(pred, calib, nbins = nbins, x_range = x_range, y_range = ratio_range_2d, title=title, offset=True)

            title = 'E_{pred}/E_{calib}^{tot} vs. E_{calib}^{tot} ' + key2 + ';(E_{calib}^{tot} + 1) [GeV];E_{pred}/E_{calib}^{tot};Count'
            x_range = [1.,1. + 0.01 * max_energy_2d]
            nbins = bins_2d
            nbins[0] = nbins[0] - 1
            mean_curves_zoomed[key][dkey], clusterE_ratio2D_zoomed[key][dkey] = EnergyPlot2D(pred, calib, nbins = nbins, x_range = x_range, y_range = ratio_range_2d, title=title, offset=False)        
        
        plots = [clusterE, clusterE_calib, clusterE_pred, clusterE_true, stacks, clusterE_ratio2D, clusterE_ratio2D_zoomed]
        if(not full): plots = [stacks, clusterE_ratio2D, clusterE_ratio2D_zoomed]
            
        dkeys = list(dsets.keys())
        
        # make legend for the overlapping plots
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
            if(plot == stacks):
                for j, dkey in enumerate(dkeys):
                    canvs[key].cd(x + j)
                    plot[key][dkey].Draw('NOSTACK HIST')
                    rt.gPad.SetLogy()
                    rt.gPad.SetGrid()
                    plot[key][dkey].GetHistogram().GetXaxis().SetTitle('E_{x}/E_{calib}^{tot}')
                    plot[key][dkey].GetHistogram().GetYaxis().SetTitle(clusterE_ratio1[key][dkey].GetYaxis().GetTitle())
                
                    if(strat == 'jet'):
                        plot[key][dkey].SetMinimum(5.0e-1)
                        plot[key][dkey].SetMaximum(1.0e3)
                    
                    else:
                        plot[key][dkey].SetMinimum(5.0e-1)
                        plot[key][dkey].SetMaximum(2.0e5)   
                    
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