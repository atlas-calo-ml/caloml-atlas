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
@jit
def Polar2Cartesian(pt,eta,phi,e):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.array([px,py,pz,e],dtype=np.dtype('f8'))

# Perform jet clustering. The "energy" parameter determines
# the name of the branch used for topo-cluster energies. If
# it is set to None (default), we use our ML energy regression.
def ClusterJets(ur_trees, jet_name, R, pt_min, eta_max, fj_dir, classification_threshold = 0.5, energy_tree_key=None, energy_branch=None, tree_name = 'JetTree', input_GeV = True, debug=False):
    
    energy_scaling = 1.
    if(input_GeV): energy_scaling = 1.0e3 # jet info saved in MeV by convention
    
    # import fastjet!
    sys.path.append(fj_dir)
    import fastjet as fj
    
    # Create our jet definition for clustering -- we use anti-kt.
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    
    # branch buffer for our jet tree, where we will save the jets we cluster
    branch_buffer = {
        jet_name + 'Pt':rt.std.vector('float')(),
        jet_name + 'Eta':rt.std.vector('float')(),
        jet_name + 'Phi':rt.std.vector('float')(),
        jet_name + 'E':rt.std.vector('float')()
    }
    
    # loop through the uproot trees (corresponding to some set of input files)
    for dfile, trees in ur_trees.items():
        
        # event info - which clusters belong to a given event
        cluster_min = trees['event']['clusterCount'].array().to_numpy()
        cluster_max = cluster_min + trees['event']['nCluster'].array().to_numpy() - 1
    
        # Reco cluster info. Excluding the energy, which will be packaged separately below.
        cluster_vec = np.column_stack(tuple(trees['cluster'].arrays(['clusterPt','clusterEta','clusterPhi','clusterE'],library='np').values()))
    
        if(energy_branch == None or energy_tree_key == None):
            # Topo-cluster classifications for all of the clusters in this file.
            cluster_classification = trees['score']['charged_likelihood_combo'].array()
            # Topo-cluster regressed energies for all clusters in this file (regressions assuming cluster comes from charged/neutral pion)
            cluster_energies = np.column_stack(tuple(trees['score'].arrays(['clusterE_charged','clusterE_neutral'],library='np').values()))
        
        else: cluster_energies = trees[energy_tree_key][energy_branch].array()

        # ROOT access to the file -- we are making a new tree to save the jet information.
        f = rt.TFile(dfile, 'UPDATE')
        t = rt.TTree(tree_name, tree_name)
        branches = {}
        for key,val in branch_buffer.items(): branches[key] = t.Branch(key, val)
        vec_polar = rt.Math.PtEtaPhiEVector() # for performing polar/Cartesian conversions for fastjet
        
        # loop over events
        nevents = trees['event'].numentries
        if(debug):
            print('File:',dfile)
            print('\tPerforming jet clustering for {val} events.'.format(val=nevents))
        for i in range(nevents):
              
            # Explicit list of cluster indices we're working with -- these are indices in ClusterTree, corresponding to event i.                
            cluster_idxs = np.array(range(cluster_min[i], cluster_max[i] + 1),dtype=np.dtype('i8'))            
            cartesian_coords = np.zeros((len(cluster_idxs),4),dtype=np.dtype('f8'))
            for j, idx in enumerate(cluster_idxs):
                
                # Energy assignment for each topo-cluster. By default, we use the cluster classifications and energy regressions
                # to assign energies, but the "energy" parameter to this function can be used to explictly pick a branch to access.
                
                if(energy_branch == None or energy_tree_key == None):
                    # Swap in the regressed energy, start off assuming this cluster corresponds with a charged pion.
                    energy = cluster_energies[idx,0]
                    # Switch to neutral energy regression if dictated by classification.
                    if cluster_classification[idx] < classification_threshold: energy = cluster_energies[idx,1]
                else: energy = cluster_energies[idx]
                    
                # If the regressed energy is infinite or nan, switch back to the reco energy.
                if(np.isinf(energy) or np.isnan(energy)): 
                    energy = cluster_vec[idx,3]
                    pt = cluster_vec[idx,0]
                    
                else:
                    # Get the ratio of the regressed energy to the original reco energy.
                    energy_ratio = energy / cluster_vec[idx,3]
            
                    # Rescale the pT according to how we've changed the topo-cluster energy (don't modify eta, phi).
                    pt = cluster_vec[idx,0] * energy_ratio

                # Create 4-vector representing the topo-cluster.
                vec_polar.SetCoordinates(pt,cluster_vec[idx,1],cluster_vec[idx,2],energy)
                cartesian_coords[j,:] = [vec_polar.Px(), vec_polar.Py(), vec_polar.Pz(), vec_polar.E()]
                #cartesian_coords[j,:] = Polar2Cartesian(pt,cluster_vec[idx,1],cluster_vec[idx,2],energy)

            # Perform jet clustering.            
            pseudojets = [fj.PseudoJet(x[0],x[1],x[2],x[3]) for x in cartesian_coords]
            jets = jet_def(pseudojets)
            # Apply optional minimum jet pT cut.
            jet_pt = np.array([jet.pt() for jet in jets])
            jet_indices = np.linspace(0,len(jets)-1,len(jets),dtype=np.dtype('i8'))[jet_pt >= pt_min]
            jets = [jets[i] for i in jet_indices]
        
            # Apply optional maximum |eta| cut.
            jet_eta = np.array([jet.eta() for jet in jets])
            jet_indices = np.linspace(0,len(jets)-1,len(jets),dtype=np.dtype('i8'))[np.abs(jet_eta) <= eta_max]
            jets = [jets[i] for i in jet_indices]
        
            # Save jet info to a TTree.
            njets = len(jets)
            for key in branch_buffer.keys(): branch_buffer[key].clear() 
            for j in range(njets):    
                branch_buffer[jet_name + 'Pt'].push_back(jets[j].pt() * energy_scaling)
                branch_buffer[jet_name + 'Eta'].push_back(jets[j].eta())
                branch_buffer[jet_name + 'Phi'].push_back(jets[j].phi())
                branch_buffer[jet_name + 'E'].push_back(jets[j].e() * energy_scaling)
        
            t.Fill()
        t.Write(tree_name, rt.TObject.kOverwrite)
        f.Close()
    return

# Returns pairs & lists of matched and unmatched indices, for a single event. These indices are w.r.t.
# whatever is given as input -- if some jets have been dropped from lists (for not passing cuts)
# this will *not* be known to jet_matching(). (i.e. it will always work 
# internally with a set of sequential indices, with which it reports results.)
def JetMatching(reco_jets, truth_jets, max_distance = 0.3):
    ntruth = len(truth_jets['eta'])
    nreco = len(reco_jets['eta'])
    reco_indices = np.linspace(0, nreco, nreco + 1, dtype = np.dtype('i2'))
    
    #TLorentzVectors for computing deltaR
    vec1 = rt.Math.PtEtaPhiEVector()
    vec2 = rt.Math.PtEtaPhiEVector()

    matched_indices = []
    unmatched_truth = []
    unmatched_reco = []
    
    for i in range(ntruth):
        truth_eta = truth_jets['eta'][i]
        truth_phi = truth_jets['phi'][i]
        vec1.SetCoordinates(0.,truth_eta,truth_phi,0.)
        
        # get distances between this truth jet and all unmatched reco jets
        distances = np.zeros(nreco)
        for j in range(nreco):
            reco_idx = reco_indices[j]
            if(reco_idx < 0):
                distances[j] = -999.
                continue 
            vec2.SetCoordinates(0.,reco_jets['eta'][reco_idx],reco_jets['phi'][reco_idx],0.)
            distances[j] = rt.Math.VectorUtil.DeltaR(vec1,vec2)
            
        # now find the minimum distance, beware of negative values
        # see https://stackoverflow.com/a/37973409
        valid_idx = np.where(distances >= 0.)[0]
        
        if(len(valid_idx) == 0):
            unmatched_truth.append(i)
            continue
        
        match_idx = valid_idx[distances[valid_idx].argmin()]
        matched_indices.append((i, match_idx))
        reco_indices[match_idx] = -1.
    unmatched_reco = reco_indices[reco_indices > -1]
    
    return {'truth_reco':matched_indices, 'unmatched_truth':unmatched_truth, 'unmatched_reco':unmatched_reco}

# Convenience function. This converts from
# indices with respect to a list of selected jets, to indices
# with respect to all indices in the original tree (passed as inputs).
@jit
def IndexConversion(index_matrix, reco_indices, truth_indices):
    index_matrix[:,0] = truth_indices[index_matrix[:,0]]
    index_matrix[:,1] = reco_indices[index_matrix[:,1]]
    return index_matrix
     
def MatchRecoJets(ur_trees, jet_defs, R, eta_max, truth_e_min, tree_name = 'JetMatchTree', input_GeV = True):

    scaling_factor = 1.0e3
    if(not input_GeV): scaling_factor = 1 # assuming MeV if not GeV
        
    reco_jet_defs = [x for x  in jet_defs.keys() if x != 'Truth']
    index_type = ('int',np.dtype('i4')) # allows us to choose how we store matching indices, e.g. int, short
    branch_buffer = {jet_defs[reco_def][1] + 'Match':rt.std.vector(index_type[0])() for reco_def in reco_jet_defs}

    for dfile, tree in ur_trees.items():   
        # tree for saving jet matching info
        f = rt.TFile(dfile, 'UPDATE')
        t = rt.TTree(tree_name, tree_name)
        branches = {}
        for key,val in branch_buffer.items():
            branches[key] = t.Branch(key, val)
    
        # Determine which jets pass our global eta cut. In practice, we have have already applied this during jet clustering itself.
        eta = {key:tree[val[0]].array(val[1] + 'Eta') for key, val in jet_defs.items()}
        jet_indices = {key: np.abs(x) <= eta_max for key,x in eta.items()}
    
        # Apply our truth jet energy cut. Recall that jets have things stored in MeV for now, whereas the cut is in GeV.
        truth_energy = tree[jet_defs['Truth'][0]][jet_defs['Truth'][1] + 'E'].array()
        jet_indices['Truth'] = jet_indices['Truth'] * (truth_energy >= scaling_factor * truth_e_min)
    
        # We will also need phi info for performing the matching.
        phi = {key:tree[val[0]][val[1] + 'Phi'].array() for key, val in jet_defs.items()}

        nevents = ur_trees[dfile]['event'].numentries
        for i in range(nevents):
        
            # For each event, store the jets that have passed our pre-selection cuts.
            selected_jets = {}
        
            for key in jet_indices.keys():
                selected_jets[key] = {'eta':eta[key][i][jet_indices[key][i]],'phi':phi[key][i][jet_indices[key][i]]}
            
            # TODO: Consider eliminating this conditional, see if it really speeds things up at all.
            # Speedup: Skip to next event if *no* truth jets were selected (there will be no matching to do anyway).
            # Make sure to fill our branch buffers with -1 for each reco jet, to explicitly label them as unmatched.
            # (We want the length of these vectors to *always* match the number of reco jets!)
            if(len(selected_jets['Truth']['eta']) == 0): 
                for rdef in reco_jet_defs:
                    buffer_key = jet_defs[rdef][1] + 'Match'
                    branch_buffer[buffer_key].clear()
                    n = len(jet_indices[rdef][i]) # total number of reco jets for this event (including those not passing cuts)
                    for j in range(n): branch_buffer[buffer_key].push_back(int(-1))
                t.Fill()
                continue
            
            # Keeping track of which indices are present and dropped.
            # For each jet definition, this dictionary lists all jet indices w.r.t. the tree, that have passed cuts.
            # E.g. if EM jets at positions 0, 4 and 5 in our branch passed cuts, then jet_tree_indices['EM']=np.array([0,4,5]).
            # More user-friendly than a boolean array of full length, e.g. [True, False, False, False, True, True,...].
            # Note that jet_indices[key] is an Awkward array, so jet_indices[key][i] gives the i'th entry, for the i'th event,
            # whose length might differ from the j'th entry jet_indices[key][j] (a.k.a. a "jagged array").
            jet_tree_indices = {key:np.linspace(0,len(jet_indices[key][i])-1,len(jet_indices[key][i]),dtype=np.dtype('i2'))[jet_indices[key][i]] for key in jet_indices.keys()}        
        
            # Now perform matching for each reco jet definition.
            for rdef in reco_jet_defs:
                buffer_key = jet_defs[rdef][1] + 'Match' # key for accessing TTree branch buffer
                branch_buffer[buffer_key].clear()

                # Info on (reco,truth) pairs, as well as any unmatched reco and truth jets.
                matches, unmatch_true, unmatch_reco = JetMatching(reco_jets=selected_jets[rdef], truth_jets=selected_jets['Truth'], max_distance=R).values()
            
                # If we don't find matches, we can skip the rest of this loop. 
                # Make sure to fill the vectors with a bunch of -1's to indicate unmatched reco jets.
                # (We want the length of these vectors to *always* match the number of reco jets!)
                if(matches == []):
                    n = len(jet_indices[rdef][i])
                    for j in range(n): branch_buffer[buffer_key].push_back(int(-1))
                    continue
                
                # Convert from indices amongst selected jets, to indices amongst all jets in tree.
                matches = IndexConversion(np.array(matches,dtype=index_type[1]), jet_tree_indices[rdef], jet_tree_indices['Truth'])           
     
                # Now prepare to fill the branch buffer. Each position correspond to a reco jet in our tree.
                # We will fill each position with the index of the truth jet (in the tree) to which it's matched.
                # If a particular reco jet is not matched, we will label it with a -1.
        
                # We prep results in a numpy array, we'll push_back the whole thing when ready.
                match_list = np.full(len(jet_indices[rdef][i]),-1,dtype=index_type[1])
                for row in matches: match_list[row[1]] = row[0]            
            
                # Now prepare to fill the branch buffer. Each position correspond to a reco jet in our tree.
                # We will fill each position with the index of the truth jet (in the tree) to which it's matched.
                # If a particular reco jet is not matched, we will label it with a -1.
            
                # Note that the first column of matches gives truth jet indices, the second column gives reco indices.
                # We will first fill the vector with -1's, then make replacements as necessary (we can treat it like np vector).
                for entry in match_list: branch_buffer[buffer_key].push_back(int(entry)) # TODO: Why do we need to cast to Python int?
            
            # The branch buffers have been filled, time to write this event to the tree.
            t.Fill()
    
        # Done writing events. Write tree to file.
        t.Write(tree_name,rt.TObject.kOverwrite)
        f.Close()
    return

def PlotEnergyRatio(ur_trees, reco_jet_defs, colors, truth_jet_def='AntiKt4TruthJets',match_key='jet_match', min_ratio = 0., max_ratio = 5., nbins = 50, paves = [], plot_dir = ''):
    
    hists = {key: rt.TH1F(str(uuid.uuid4()), key + ';E_{reco}/E_{true};Count',nbins,min_ratio,max_ratio) for key in reco_jet_defs.keys()}
    hists_zoomed = {key: rt.TH1F(str(uuid.uuid4()), key + ' (zoomed axis);E_{reco}/E_{true};Count',10 * nbins,min_ratio,max_ratio) for key in reco_jet_defs.keys()}

    for key in reco_jet_defs.keys():
        qu.SetColor(hists[key],colors[key])
        qu.SetColor(hists_zoomed[key],colors[key])

    # now loop through our files and fill the trees
    for dfile, tree in ur_trees.items():
        truth_energy = tree['event'][truth_jet_def + 'E'].array()
    
        # make a histogram for each reco jet definition
        for key, rdef in reco_jet_defs.items():
            reco_energy = tree[rdef[0]][rdef[1] + 'E'].array()
            reco_match  = tree['jet_match'][rdef[1] + 'Match'].array()
        
            # looping through events in this file (these are jagged arrays)
            for i in range(len(reco_match)):
                # looping through truth jets, plot match for each one that has a match
                for j in range(len(reco_match[i])):
                    if(reco_match[i][j] > -1): 
                        hists[key].Fill(reco_energy[i][j] / truth_energy[i][reco_match[i][j]])
                        hists_zoomed[key].Fill(reco_energy[i][j] / truth_energy[i][reco_match[i][j]])

    # draw the histograms we just made
    c = rt.TCanvas(str(uuid.uuid4()),'c_energy_ratio',800, 1200)
    rt.gStyle.SetOptStat(0)

    legend = rt.TLegend(0.7,0.7,0.9,0.9)
    legend.SetTextColor(rt.gStyle.GetTextColor())
    for key in hists.keys(): legend.AddEntry(hists[key],key,'f')

    c.Divide(1,2)
    c.cd(1)
    stack1 = rt.THStack(str(uuid.uuid4()),'s1')
    stack1.SetTitle('Jet energy ratio;E_{reco}/E_{true};Count')
    for hist in hists.values(): stack1.Add(hist)
    stack1.Draw('NOSTACK HIST')
    rt.gPad.SetLogy()
    legend.Draw()
    for pave in paves: pave.Draw()


    c.cd(2)
    stack2 = rt.THStack(str(uuid.uuid4()),'s1')
    stack2.SetTitle('Jet energy ratio (zoomed axis);E_{reco}/E_{true};Count')
    for hist in hists_zoomed.values(): stack2.Add(hist)
    stack2.Draw('NOSTACK HIST')
    rt.gPad.SetLogx()
    rt.gPad.SetLogy()
    legend.Draw()
    for pave in paves: pave.Draw()

    c.SaveAs(plot_dir + '/' + 'jet_energy_ratio.png')
    c.Draw()
    
    # return the ROOT objects, so that things display on the canvas
    results = {}
    results['canvas'] = c
    results['stacks'] = [stack1, stack2]
    results['legend'] = legend
    results['hists'] = [hists,hists_zoomed]
    return results

def PlotJetKinematics(ur_trees, jet_defs, colors, plot_dir, eta_max, truth_e_min, paves = [], logx = [], input_GeV=True):
    
    scale_factor = 0.001 # jet info is in MeV, we want to plot it all in GeV
    unit = 'GeV'
    if(not input_GeV): 
        scale_factor = 1. # if not GeV, assume MeV
        unit = 'MeV'

    energy_hists = {key:rt.TH1F(str(uuid.uuid4()), key + ' Jets;Energy [{val}];Count'.format(val=unit), 60, 0., 300.) for key in jet_defs.keys()}
    pt_hists     = {key:rt.TH1F(str(uuid.uuid4()), key + ' Jets;p_{T}' + ' [{val}];Count'.format(val=unit), 60, 0., 300.) for key in jet_defs.keys()}
    eta_hists    = {key:rt.TH1F(str(uuid.uuid4()), key + ' Jets;#eta;Count', 50, -1., 1.) for key in jet_defs.keys()}
    m_hists      = {key:rt.TH1F(str(uuid.uuid4()), key + ' Jets;m [{val}];Count'.format(val=unit), 55, -10., 100.) for key in jet_defs.keys()}
    ep_hists     = {key:rt.TH1F(str(uuid.uuid4()), key + ' Jets;Energy / p_{T};Count', 90, 0.9, 1.2) for key in jet_defs.keys()}
    n_hists      = {key:rt.TH1I(str(uuid.uuid4()), key + ' Jets;N_{jets};Count', 10, 0, 10) for key in jet_defs.keys()}

    vec = rt.Math.PtEtaPhiEVector()
    for dfile, tree in ur_trees.items():
        for key, jet_def in jet_defs.items():
            tkey  = jet_def[0]
            jname = jet_def[1]
        
            # Truth jets -- apply global jet cuts, and truth-specific jet cuts.
            if(key == 'Truth'):
                eta = tree[tkey].array(jname + 'Eta')
                energy = tree[tkey].array(jname + 'E')
                # Truth jet energy cut & jet eta cut.
                jet_indices = (np.abs(eta) <= eta_max) * (scale_factor * energy  >= truth_e_min)
        
            # Reco jets -- apply jet-matching cut (global jet cuts are built-in). No further reco-specific jet cuts for now.
            else:
                matching = tree['jet_match'].array(jname + 'Match')
                jet_indices = (matching > -1)
            
            # Now get all jets that passed the cuts.
            # First do a bit of manipulation with eta, to also get the number of jets per event.
            eta = tree[tkey].array(jname + 'Eta')[jet_indices]
            n   = [len(x) for x in eta]
            eta = eta.flatten()
            energy = scale_factor * tree[tkey].array(jname + 'E')[jet_indices].flatten()
            pt     = scale_factor * tree[tkey].array(jname + 'Pt')[jet_indices].flatten()
            ep     = energy / pt
        
            for i in range(len(n)): n_hists[key].Fill(n[i])
            
            for i in range(len(ep)):
                energy_hists[key].Fill(energy[i])
                pt_hists[key].Fill(pt[i])
                eta_hists[key].Fill(eta[i])
                ep_hists[key].Fill(ep[i])
            
                # Compute the jet mass and plot it too.
                vec.SetCoordinates(pt[i],eta[i],0.,energy[i])
                m_hists[key].Fill(vec.M())

    hist_lists = [energy_hists, pt_hists, eta_hists, ep_hists, m_hists, n_hists]
    names      = ['energy',     'pt',     'eta',      'ep',    'm',     'n']
    for key in jet_defs.keys():
        for hist_list in hist_lists:
            alpha = 0.5
            if(key == 'Truth'): alpha = 0.75
            qu.SetColor(hist_list[key],colors[key],alpha = alpha)            
            
    rt.gStyle.SetOptStat(0)
    canvases = []

    for i in range(len(hist_lists)):
        hist_list = hist_lists[i]
        use_logx = False
        if(names[i] in logx): use_logx = True
        c = qu.DrawSet(hist_list, logx=use_logx, paves = paves)
        canvases.append(c)
        c.Draw()
        c.SaveAs(plot_dir + '/' + names[i] + '.png')
        
    results = {}
    results['canvas'] = canvases
    results['hists'] = hist_lists
    return results