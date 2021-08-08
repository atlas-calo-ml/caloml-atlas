/*
 * ROOT/C++ macro
 * We are performing jet matching, using output from TrigL0GepPerf.
 * The jet matching code is adapted from IsolatedJetTree, the difference
 * here is that we are not running on an AOD file
 * (and thus lack certain information and running options).
 */

// ROOT includes
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TString.h"
#include "TObject.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TList.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"
#include "TMath.h"

// standard model includes
#include <utility> //std::pair
#include <vector> // std::vector
#include <map> // std::map
#include <queue> // std::priority_queue
#include <algorithm> // std::max_element
#include <iostream> // std::cout (for debugging)
#include <numeric> // std::iota
#include <string>
#include <stdio.h> // printf
#include <stdlib.h>

// Global variables
std::vector<TString> coord_names = {"Pt", "Eta", "Phi", "E"};


// Helper function, returns Nth largest element in vector.
// see: https://cppsecrets.com/users/41129711010797106994610011511264103109971051084699111109/Find-the-Nth-largest-element-in-a-vector.php
Float_t getNthLargestElement(vector<Float_t>& v, 
                             UInt_t nthLargestElement){
    std::priority_queue<Float_t, vector<Float_t>, std::greater<Float_t>> minHeap;
    for(UInt_t i = 0; i < v.size(); i++) {
        if(i < nthLargestElement) {
            minHeap.push(v[i]);
        } else {
            // minHeap.top() will reaturn minimum
            // element of the priority queue
            if(v[i] > minHeap.top()) {
                // Remove minimum element
                minHeap.pop(); 
                minHeap.push(v[i]);
            }
        }
    }
        return minHeap.top();
}

// Returns a list of jet indices for jets that pass a set of cuts
// (cuts on eta, phi, energy)
std::vector<Int_t> applyJetCuts(Float_t pt_min, 
                                Float_t eta_max,
                                std::map<TString, TTreeReaderArray<Double_t>*> r_vec){
    std::vector<Int_t> jet_indices; // will store the indices of jets that pass the cuts (indices w.r.t. vector in branch)
    ROOT::Math::PtEtaPhiEVector* vec = new ROOT::Math::PtEtaPhiEVector(0.,0.,0.,0.); // LorentzVector for easy coordinate access
//     Int_t njets = r_vec["pt"]->GetSize();
    Int_t njets = r_vec[coord_names.at(0)]->GetSize();
    for(Int_t i = 0; i < njets; i++){
        vec->SetCoordinates(r_vec[coord_names.at(0)]->At(i), r_vec[coord_names.at(1)]->At(i), r_vec[coord_names.at(2)]->At(i), r_vec[coord_names.at(3)]->At(i));

        // No negative-energy jets. (Cut is applied in IsolatedJetTree/Root/IsolatedJetAlgo.cxx)
        if(vec->E() < 0.) continue;
        
        // Pt and eta cuts. (Cuts are applied by JetSelector, see IsolatedJetTree/data/config_Tree.py)
        if(vec->Pt() < pt_min || TMath::Abs(vec->Eta()) > eta_max) continue;
        jet_indices.push_back(i);
    }
    return jet_indices;
}

// Checks if set of reco and truth jets pass truth pileup requirement.
// Based on code within IsolatedJetTree/Root/IsolatedJetAlgo.cxx::execute()
Bool_t truthPileupCheck(std::vector<Int_t> reco_indices,
                        std::vector<Int_t> truth_indices,
                        std::map<TString, TTreeReaderArray<Double_t>*> r_reco,
                        std::map<TString, TTreeReaderArray<Double_t>*> r_truth){
    std::vector<Float_t> reco_pt; //  access pt of reco  jets passing cuts
    std::vector<Float_t> truth_pt; // access pt of truth jets passing cuts
    for(Int_t j : reco_indices) reco_pt.push_back(r_reco[coord_names.at(0)]->At(j));
    for(Int_t j : truth_indices) truth_pt.push_back(r_truth[coord_names.at(0)]->At(j));
    Float_t pt_avg = (getNthLargestElement(reco_pt, 1) + getNthLargestElement(reco_pt, 2))/2.0; // getNthLargestElement uses 1-indexing (1st largest == largest)
    if(pt_avg / getNthLargestElement(truth_pt, 1) > 1.4) return kFALSE;
    return kTRUE;
}

// Checks if a given jet is isolated.
// Based on IsolatedJetTree/IsolatedJetTree/IsolationHelpers.h::isIsolated()
Bool_t isIsolated(Int_t this_jet_index,
                  std::vector<Int_t> jet_indices,
                 std::map<TString, TTreeReaderArray<Double_t>*> r_jet,
                 Float_t dr2_cut){
    Bool_t isolated = kTRUE;
    // declare two 4-vecs we will use for comparison,
    // one for this jet (this_vec) and one for the other jets (that_vec)
    ROOT::Math::PtEtaPhiEVector* this_vec = new ROOT::Math::PtEtaPhiEVector(0.,0.,0.,0.);
    ROOT::Math::PtEtaPhiEVector* that_vec = new ROOT::Math::PtEtaPhiEVector(0.,0.,0.,0.);

    // Set up 4-vector for this jet. Note that only eta and phi are needed for this matching.
    this_vec->SetCoordinates(0.,r_jet[coord_names.at(1)]->At(this_jet_index),r_jet[coord_names.at(2)]->At(this_jet_index),0.);
    
    for(Int_t idx: jet_indices){
        if(idx == this_jet_index) continue;
        that_vec->SetCoordinates(0.,r_jet[coord_names.at(1)]->At(idx),r_jet[coord_names.at(2)]->At(idx),0.);
        Double_t dr2 = ROOT::Math::VectorUtil::DeltaR2(*this_vec, *that_vec);
        if(dr2 < dr2_cut){
            isolated = kFALSE;
            break;
        }
    }
    return isolated;
}

// For a given jet, returns index of matched jet with respect to the list
// of selected jets (jet_indices). Returns -1 if no match is found.
Int_t getMatch(Int_t this_jet_index,
               std::vector<Int_t> matching_jet_indices,
               std::map<TString, TTreeReaderArray<Double_t>*> r_this_jet,
               std::map<TString, TTreeReaderArray<Double_t>*> r_matching_jet,
               Float_t dr2_cut){
    Int_t matchedIndex = -1;
    std::vector<Float_t> dr2_vec;
    ROOT::Math::PtEtaPhiEVector* this_vec = new ROOT::Math::PtEtaPhiEVector(0.,0.,0.,0.);
    ROOT::Math::PtEtaPhiEVector* that_vec = new ROOT::Math::PtEtaPhiEVector(0.,0.,0.,0.);
    
//     std::cout << "\t\t\t[GM] " << matching_jet_indices.size() << std::endl; //dbg
    
    // Set up 4-vector for this jet. Note that only eta and phi are needed for this matching.
    this_vec->SetCoordinates(0.,r_this_jet[coord_names.at(1)]->At(this_jet_index),r_this_jet[coord_names.at(2)]->At(this_jet_index),0.);
    
    for(Int_t idx: matching_jet_indices){
        that_vec->SetCoordinates(0.,r_matching_jet[coord_names.at(1)]->At(idx),r_matching_jet[coord_names.at(2)]->At(idx),0.);
        dr2_vec.push_back(ROOT::Math::VectorUtil::DeltaR2(*this_vec, *that_vec));
    }
        
    // Get the jet index corresponding to the minimum dR2
    Int_t min_dr2_idx = std::min_element(dr2_vec.begin(),dr2_vec.end()) - dr2_vec.begin();
    Double_t min_dr2 = dr2_vec.at(min_dr2_idx);
    if(min_dr2 < dr2_cut) matchedIndex = min_dr2_idx;
//     if(min_dr2 < dr2_cut) matchedIndex = matching_jet_indices.at(min_dr2_idx);
    return matchedIndex;    
}

/*
 * Given an ntuple (from TrigL0GepPerf or this function itself),
 * create a copy where we add a TTree with information on jet matching.
 */
void matchNtuple(TString trigger_jet_file = "../../data/tl0gp/myfile_tree.root", 
                 TString output_file = "test.root", 
                 TString tree_name = "MatchTree",
                 TString reco_jet_name = "AntiKt4emtopoCalo422Jets",
                 TString truth_jet_name = "AntiKt4lctopoCaloCalJets",
                 Float_t pt_min = 7.0e3,
                 Float_t eta_max = 4.5,
                 Bool_t requirePileupCheck = kFALSE,
                 Bool_t requireIsoReco = kTRUE,
                 Bool_t requireIsoTruth = kTRUE,
                 Float_t truth_dr = 0.3,
                 Float_t truth_iso_dr = 0.3,
                 Float_t reco_iso_dr = 0.3){
    /* 
     * Setup: We need to define the ""reco" and "truth" jets for matching.
     * Note that in practice, the "truth" jets might actually be reco jets,
     * as we will match trigger jets to (standard) reco jets (and reco jets to truth jets).
     * Both sets of jets will be fetched from our trigger jet file, which should contain branches
     * containing information on offline reco jets, our trigger jets, and truth jets.
     *
     * Available jets:
     * Trigger: AntiKt4emtopoCalo422Jets, AntiKt4emtopoCalo422SKJets, AntiKt4emtopoCalo422VorSKJets
     * Reco   : AntiKt4lctopoCaloCalJets, AntiKt4lctopoCaloCalSKJets, AntiKt4lctopoCaloCalVorSKJets
     * Truth  : TODO (not added yet)
     */
    Float_t truth_dr2 = truth_dr * truth_dr;
    Float_t truth_iso_dr2 = truth_iso_dr * truth_iso_dr;
    Float_t reco_iso_dr2 =  reco_iso_dr * reco_iso_dr;

    // Fetch the input TTree
    TFile* f_jet = new TFile(trigger_jet_file,"READ");
    TTree* jet_tree = (TTree*)f_jet->Get("ntuple");
    /*
     * Preparing the output ntuple:
     *
     * The output of this routine will be a new TTree with two branches
     * of type std::vector<Int_t>, which will have matching information
     * for the two species of jets we are matching.
     */
    TFile* f_out = new TFile(output_file, "RECREATE");
    
    // copy all trees from input to output
    TList* file_keys = f_jet->GetListOfKeys();
    for (TObject* obj: *file_keys){
        TString key = ((TObjString*)obj)->String();
        TTree* t = nullptr;
        f_jet->GetObject(key, t);
        if(t){
            f_out->cd();
            TTree* t_copy = t->CloneTree();
            t_copy->Write();
        }
    }
    
    TTree* t_out = new TTree(tree_name, tree_name);
    std::vector<Int_t> reco_status;
    std::vector<Int_t> truth_status;
    t_out->Branch("RecoJetStatus", &reco_status);
    t_out->Branch("TruthJetStatus", &truth_status);
    
    // Prepare some meta-data for t_out, explaining which two species of jets it refers to.
    TString matching_metadata = reco_jet_name + "->" + truth_jet_name;
    TObjString* metadata = new TObjString(matching_metadata);
    t_out->GetUserInfo()->Add(metadata);
        
    // Setup reader. We use maps of pointers to readers to compactly access the 4-vectors (whose components are split across branches).
    TTreeReader jtReader(jet_tree);
    std::map<TString, TTreeReaderArray<Double_t>*> r_reco;  // for reading reco jet 4-vector components
    std::map<TString, TTreeReaderArray<Double_t>*> r_truth; // for reading truth jet 4-vector components
    for (TString coord: coord_names){
        r_reco.insert(std::pair<TString,  TTreeReaderArray<Double_t>*>(coord, new TTreeReaderArray<Double_t>(jtReader, reco_jet_name + coord)));
        r_truth.insert(std::pair<TString, TTreeReaderArray<Double_t>*>(coord, new TTreeReaderArray<Double_t>(jtReader, truth_jet_name + coord)));
    }    

    Long64_t nentries = jet_tree->GetEntries();
    // Event loop.
    for(Long64_t i = 0; i < nentries; i++){
        // Clear our branch buffers.
        reco_status.clear();
        truth_status.clear();
//         std::cout << "i = " << i << std::endl; //dbg
        
        // Additional (local) buffers for keeping track of indices of jets we match for this event.
        // These will later be used to fill in branch_buffer.
        std::vector<Int_t> matched_reco_indices;
        std::vector<Int_t> matched_truth_indices;

        jtReader.SetEntry(i);
        Bool_t skip_event = kFALSE;
        // First, we must gather the truth and reco jets we are matching.
        // We'll do this by keeping track of the indices of jets in the input ntuple.
        // We will apply some preliminary cuts to each category.
        std::vector<Int_t> reco_indices  = applyJetCuts(pt_min, eta_max, r_reco );
        std::vector<Int_t> truth_indices = applyJetCuts(pt_min, eta_max, r_truth);
//         std::cout << "Number of reco/truth jets passing cuts: " << reco_indices.size() << "/" << truth_indices.size() << std::endl; // dbg
        // Begin Isolated Jet selections
        // at least 1 truth jet and 2 reco jets
        if(truth_indices.size() == 0 || reco_indices.size() <= 1){
            skip_event = kTRUE;
//             std::cout << "\t\t Skip (#)" << std::endl; //dbg
        }
        
        // truth pile-up check
        if(!skip_event && requirePileupCheck && !truthPileupCheck(reco_indices, truth_indices, r_reco, r_truth)){
            skip_event = kTRUE;
//             std::cout << "\t\t Skip (pileup)" << std::endl; //dbg
        }
        // Now perform jet matching, with optional isolation requirement on truth and reco jets.
        // We loop over reco jets, as in IsolatedJetTree/Root/IsolatedJetAlgo.cxx::execute().
        // Note that this loop is done in order of decreasing pT, so we start w/ the leading reco jets.
        
        Int_t n_match = 0;
        if(!skip_event){
            // To make sure that we are always looping in order of decreasing pT, we will explicitly
            // perform a pT sorting here. This is useful in case, for whatever reason, this code
            // is run on an ntuple where we've already done matching and the pT ordering is lost.
            std::vector<Float_t> reco_pt;
            for (Int_t reco_idx: reco_indices) reco_pt.push_back(r_reco[coord_names.at(0)]->At(reco_idx));
            std::vector<Int_t> reco_pt_sorting(reco_indices.size());
            std::iota(reco_pt_sorting.begin(), reco_pt_sorting.end(), 0);
            std::sort(reco_pt_sorting.begin(), reco_pt_sorting.end(), [&](Int_t a, Int_t b) { return reco_pt.at(a) > reco_pt.at(b); });
            std::vector<Int_t> reco_indices_sorted;
            for (Int_t idx: reco_pt_sorting) reco_indices_sorted.push_back(reco_indices.at(idx));
            
            for (Int_t reco_idx: reco_indices_sorted){
                if(truth_indices.size() == 0) continue;
                // If requested, check isolation on this reco jet.
                if(requireIsoReco && !isIsolated(reco_idx, reco_indices, r_reco, reco_iso_dr2)){
                    // Failed reco isolation requirement.
//                     std::cout << "\t\t\tFailed reco isolation requirement." << std::endl; //dbg
                    continue;
                }
                
                // Find a matching truth jet for this reco jet, if it exists.
                // This function returns the position of the truth jet index within
                // the passed vector<Int_t> of truth jet indices.
                Int_t matching_idx_loc = getMatch(reco_idx, truth_indices, r_reco, r_truth, truth_dr2);
                if(matching_idx_loc < 0){
                    // Failed to find matching truth jet.
                    //std::cout << "\t\t\tNo match. p_reco = ()" << std::endl; //dbg
                    
//                     printf("\t\t\tNo match. p_reco = (%4.2f,%4.2f,%4.2f,%4.2f)\n", 
//                            r_reco[coord_names.at(0)]->At(reco_idx),
//                            r_reco[coord_names.at(1)]->At(reco_idx),
//                            r_reco[coord_names.at(2)]->At(reco_idx),
//                            r_reco[coord_names.at(3)]->At(reco_idx)
//                           );
                    
                    
                    continue;
                }
                
                Int_t matching_idx = truth_indices.at(matching_idx_loc);

                // If requested, check isolation on the truth jet to which we just matched this reco jet.
                if(requireIsoTruth && !isIsolated(matching_idx, truth_indices, r_truth, truth_iso_dr2)){
                    // Failed truth isolation requirement
//                     std::cout << "\t\t\tFailed truth isolation requirement." << std::endl; //dbg
                    continue;
                }
                
//                 printf("\t\t\tMatch: %i -> %i \t (%4.2f,%4.2f,%4.2f,%4.2f) -> (%4.2f,%4.2f,%4.2f,%4.2f)\n",
//                        reco_idx, matching_idx,
//                        r_reco[coord_names.at(0)]->At(reco_idx),
//                        r_reco[coord_names.at(1)]->At(reco_idx),
//                        r_reco[coord_names.at(2)]->At(reco_idx),
//                        r_reco[coord_names.at(3)]->At(reco_idx),
//                        r_truth[coord_names.at(0)]->At(matching_idx),
//                        r_truth[coord_names.at(1)]->At(matching_idx),
//                        r_truth[coord_names.at(2)]->At(matching_idx),
//                        r_truth[coord_names.at(3)]->At(matching_idx)
//                       );
                
                //std::cout << "\t\t\tMatch: " << reco_idx << " -> " << matching_idx <<  " ." << std::endl; //dbg

                
                // We have found a matching truth jet & passed any isolation requirements.
                // Now we want to record the pair of jets' kinematics, to write to the output ntuple,
                // and then we want to explicitly remove the matched truth jet's index from our vector
                // so that we don't accidentally match it a 2nd time.
                matched_reco_indices.push_back(reco_idx);
                matched_truth_indices.push_back(matching_idx);
                truth_indices.erase(truth_indices.begin() + matching_idx_loc);
//                 if(truth_indices.size() == 0) break; // if we have matched every reference jet, we are done
            }
            
            /*
             * At this point we have two lists of indices:
             * - Indices of matched reco jets.
             * - Indices of matched truth jets.
             * The two lists line up, so that the i'th matched reco jet is matched to the i'th matched truth jet.
             * Any missing indices correspond to jets that failed some cut or were not matched.
             * We now want to construct a list for *all* reco and truth jets in this event, where for each
             * we provide either the index of its partner, or a -1 if it was not matched or failed a cut.
             */
            
            n_match = matched_reco_indices.size();
            for(UInt_t j = 0; j < r_reco[coord_names.at(0)]->GetSize(); j++) reco_status.push_back(-1);
            for(UInt_t j = 0; j < r_truth[coord_names.at(0)]->GetSize(); j++) truth_status.push_back(-1);            
            for(Int_t j = 0; j < n_match; j++){
                reco_status.at( matched_reco_indices.at(j) ) = matched_truth_indices.at(j);
                truth_status.at(matched_truth_indices.at(j)) = matched_reco_indices.at( j);
            }
        }
        // Event was skipped (due to too few jets or pileup check fail).
        // We will mark this by inserting statuses of "-2" instead of "-1".
        else{
            for(UInt_t j = 0; j < r_reco[coord_names.at(0)]->GetSize(); j++) reco_status.push_back(-2);
            for(UInt_t j = 0; j < r_truth[coord_names.at(0)]->GetSize(); j++) truth_status.push_back(-2); 
        }
        t_out->Fill();
    }
    t_out->Write();
    f_out->Close();
    f_jet->Close();
    return;
}

/*
 * Given an ntuple with a tree of matching info,
 * use that matching info to select and order jets.
 */
void makeMatchedNtuple(TString input_file="test.root",
                       TString output_file="test2.root",
                       TString matching_tree="MatchTree"){
    TFile* f = new TFile(input_file, "READ");
    
    // Get the ntuple tree, matching tree and any other trees present.
    TTree* jet_tree = (TTree*)f->Get("ntuple");
    TTree* match_tree = (TTree*)f->Get(matching_tree);
    
    // Prepare the output ntuple.
    TFile* g = new TFile(output_file,"RECREATE");
        
    // Copy over any trees present besides the ntuple and matching tree.
    TList* file_keys = f->GetListOfKeys();
    for (TObject* obj: *file_keys){
        TString key = ((TObjString*)obj)->String();
        if(key.EqualTo("ntuple") || key.EqualTo(matching_tree)) continue;
        TTree* t = nullptr;
        f->GetObject(key, t);
        if(t){
            TTree* t_copy = t->CloneTree();
            t_copy->Write();
        }
    }
    
    // Determine which jets are being matched. Jet_a is being matched to Jet_b.
    TString matching_string = ((TObjString*)match_tree->GetUserInfo()->At(0))->String();
    TObjArray* matching_string_split = matching_string.Tokenize("->");
    TString jet_a = ((TObjString*)matching_string_split->At(0))->String();
    TString jet_b = ((TObjString*)matching_string_split->At(1))->String();
    //delete matching_string_split;
        
    // Copy over the ntuple, but don't copy the branches that are subject to matching.
    jet_tree->SetBranchStatus("*",1); // redundant
    jet_tree->SetBranchStatus(jet_a + "*", 0);
    jet_tree->SetBranchStatus(jet_b + "*", 0);
    TTree* jet_tree_copy = jet_tree->CloneTree();
    jet_tree->SetBranchStatus("*",1);
    
    // Setup reading for jet_tree and match_tree.
    // We only need to read the branches of jets being matched.
    // TODO: We also want to read branches of any jets that have previously been matched.
    // E.g. if we are matching trigger to reco, and we've already matched reco to truth, and we
    // end up with any reco jets without truth matches, we want to throw out those reco jets
    // *and* the truth jet we previously matched them to.
    TTreeReader jtReader(jet_tree);    
    std::map<TString, TTreeReaderArray<Double_t>*> r_a;  // for reading species-a jet 4-vector components
    std::map<TString, TTreeReaderArray<Double_t>*> r_b;  // for reading species-b jet 4-vector components

    for (TString coord: coord_names){
        r_a.insert(std::pair<TString, TTreeReaderArray<Double_t>*>(coord, new TTreeReaderArray<Double_t>(jtReader, jet_a + coord)));
        r_b.insert(std::pair<TString, TTreeReaderArray<Double_t>*>(coord, new TTreeReaderArray<Double_t>(jtReader, jet_b + coord)));
    }    
    
    TTreeReader mtReader(match_tree);
    TTreeReaderArray<Int_t> r_a_stat(mtReader, "RecoJetStatus" );
    TTreeReaderArray<Int_t> r_b_stat(mtReader, "TruthJetStatus");

    // Add branches to the copy of jet_tree, for the jets being matched.
    std::map<TString, std::vector<Float_t>> jet_buffer_a;
    std::map<TString, std::vector<Float_t>> jet_buffer_b;
    std::map<TString, TBranch*> jet_branches_a;
    std::map<TString, TBranch*> jet_branches_b;
    
    std::map<TString, TString> branch_names_a;
    std::map<TString, TString> branch_names_b;

    for (TString coord: coord_names){
        TString name_a = jet_a + coord;
        TString name_b = jet_b + coord;
        
        // We declare entries in jet buffers just for the sake of passing the addresses to the TBranch's,
        // these will in principle be re-assigned anyway
        jet_buffer_a.insert(std::pair<TString, std::vector<Float_t>>(coord, std::vector<Float_t>()));
        jet_buffer_b.insert(std::pair<TString, std::vector<Float_t>>(coord, std::vector<Float_t>()));
        
        TBranch* branch_a = jet_tree_copy->Branch(name_a, &(jet_buffer_a[coord]));
        TBranch* branch_b = jet_tree_copy->Branch(name_b, &(jet_buffer_b[coord]));
        
        jet_branches_a.insert(std::pair<TString, TBranch*>(name_a,branch_a));
        jet_branches_b.insert(std::pair<TString, TBranch*>(name_b,branch_b));
        
        branch_names_a[name_a] = coord;
        branch_names_b[name_b] = coord;
    }

    Long64_t nentries = jet_tree->GetEntries();
    
    for(Long64_t i = 0; i < nentries; i++){
        jtReader.SetEntry(i);
        mtReader.SetEntry(i);
        
        // Clear our buffers. Using C++11 standard here! TODO: Is this safe?
        for(auto &entry : jet_buffer_a) entry.second.clear();
        for(auto &entry : jet_buffer_b) entry.second.clear();
        
        // Determine how many jets we're dealing with.
        UInt_t n_a = r_a[coord_names.at(0)]->GetSize();
        UInt_t n_b = r_b[coord_names.at(0)]->GetSize();
        
        for(UInt_t j = 0; j < n_b; j++){    
            Int_t entry = r_b_stat.At(j);
            if(entry < 0) continue;
            
            // record the b-species jet's 4-vector
            for(TString coord : coord_names) jet_buffer_b[coord].push_back(r_b[coord]->At(j));
            
            // record the corresponding a-species jet's 4-vector
            for(TString coord : coord_names) jet_buffer_a[coord].push_back(r_a[coord]->At(entry));
        }        
        for (auto &bname: branch_names_a) jet_branches_a[bname.first]->Fill();        
        for (auto &bname: branch_names_b) jet_branches_b[bname.first]->Fill();
    }
    jet_tree_copy->Write();
    g->Close();
    f->Close();
    //delete file_keys;
    return;
}

void JetMatching(TString input_file="infile.root", 
                 TString output_file="outfile.root", 
                 TString matching_string="AntiKt4emtopoCalo422Jets->AntiKt4lctopoCaloCalJets",
                 TString matching_settings_string="7.0e3,4.5,0,1,1,0.3,0.3,0.3",
                 TString filename_tmp = ""){
    
    /*
     * The way we have set up our matching, it is best to perform a match and apply it before
     * doing a 2nd match. This is because if we match jet_a to jet_b, then when we apply the matching
     * the ordering of jet_a will change (the ordering of jet_b will remain fixed). As the way we
     * document matching depends on the ordering, this may scramble future matches if we have
     * already queued them up with matchNtuple().
     */
    
    // The matching_string can hold a queue of matches, we will parse it.
    // If, for example, we are matching jet_b to jet_a, and then jet_c to jet_b,
    // the string should be "jet_b->jet_a;jet_c->jet_b". In general, the ordering can matter.

    if(filename_tmp.EqualTo("")) filename_tmp = TString(output_file).ReplaceAll(".root","_tmp.root");
    
    TObjArray* matches = matching_string.Tokenize(";");
    TObjArray* matching_settings = matching_settings_string.Tokenize(";");
    
    Long64_t n_matches = matches->GetEntries();
    if(matching_settings->GetEntries() != n_matches){
        std::cout << "Error: Matching settings not supplied for all matches." << std::endl;
        return;
    }
    
    // Set up the sequence of filenames. We will delete temporary files, but will
    // make sure not to delete the input or final output.
    std::vector<TString> file_names;
    file_names.push_back(input_file);
    for(Long64_t i = 0; i < n_matches; i++){
        TString tmp_name = "tmp_" + std::to_string(i) + "a.root";
        file_names.push_back(tmp_name);
        if(i == n_matches-1) file_names.push_back(output_file);
        else{
            tmp_name = "tmp_" + std::to_string(i) + "b.root";
            file_names.push_back(tmp_name);
        }
    }
    
    for(Long64_t i = 0; i < n_matches; i++){
        
        TString match_string = ((TObjString*)matches->At(i))->String();
        TString setting_string = ((TObjString*)matching_settings->At(i))->String();
                
        // Get the jets being matched
        TObjArray* jets = match_string.Tokenize("->");
        TString jet_a = ((TObjString*)jets->At(0))->String();
        TString jet_b = ((TObjString*)jets->At(1))->String();
        
        // Get the matching settings.
        TObjArray* settings = setting_string.Tokenize(",");
        if(settings->GetEntries() < 8){
            std::cout << "\tError for match sequence " << i << ", need 8 matching settings but only found " << settings->GetEntries() << std::endl;
            continue;
        }
        Float_t pt_min            = atof(((TObjString*)settings->At(0))->String()); // min pT cut
        Float_t eta_max           = atof(((TObjString*)settings->At(1))->String()); // max |eta| cut
        Bool_t reqPUCheck         = atoi(((TObjString*)settings->At(2))->String()); // do pileup check on jet_a?
        Bool_t reqIsoReco         = atoi(((TObjString*)settings->At(3))->String()); // require isolation for jet_a?
        Bool_t reqIsoTruth        = atoi(((TObjString*)settings->At(4))->String()); // require isolation for jet_b?
        Float_t dr                = atof(((TObjString*)settings->At(5))->String()); // dR used for matching jet_a & jet_b
        Float_t truth_iso_dr      = atof(((TObjString*)settings->At(6))->String()); // dR used for isolation of jet_a
        Float_t reco_iso_dr       = atof(((TObjString*)settings->At(7))->String()); // dR used for isolation of jet_b
        
        TString name0 = file_names.at(2 * i); // name of input
        TString name1 = file_names.at(2 * i + 1); // name of temp output
        TString name2 = file_names.at(2 * i + 2); // name of final output of this sequence
        
        // Compute matching info
        matchNtuple(name0, name1, "MatchTree",jet_a,jet_b, pt_min, eta_max, reqPUCheck, reqIsoReco, reqIsoTruth, dr, truth_iso_dr, reco_iso_dr); 
        if(i != 0) gSystem->Unlink(name0);  // Delete input unless it is the original input file.
        makeMatchedNtuple(name1, name2, "MatchTree");
        gSystem->Unlink(name1);
    }

    return;
}
