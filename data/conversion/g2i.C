// Graph-to-Image
// ROOT/C++ macro

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TString.h"
#include "TList.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TMatrixT.h"
#include "TMatrixFfwd.h"
#include <map>
#include <vector>

void g2i(TString infile="", TString outfile="", Float_t eta_range=0.4, Float_t phi_range=0.4){
    
    if(infile.EqualTo("") || outfile.EqualTo("")){
        std::cout << "Error: Input and/or output not specified." << std::endl;
        return;
    }
        
    Float_t eta_min = -1. * eta_range/2.0;
    Float_t phi_min = -1. * phi_range/2.0;
    
    // sampling layers (values corresponding to EMB1, EMB2, etc.)
    std::vector<TString> layers{"EMB1","EMB2","EMB3","TileBar0","TileBar1","TileBar2"};
    std::vector<UShort_t> sampling_layers{1,2,3,12,13,14};
    std::vector<UInt_t> len_eta{128,16,8,4,4,2};
    std::vector<UInt_t> len_phi{4,16,16,4,4,4};
    
    // images -- we use TMatrixF as a convenient container
    map<Int_t, TMatrixF> calo_images;
    
    for(UInt_t i = 0; i < sampling_layers.size(); i++){
        calo_images.insert(pair<Int_t, TMatrixF>(sampling_layers.at(i), TMatrixF(len_eta.at(i), len_phi.at(i))));
    }
        
    TFile* f = new TFile(infile,"READ");
    TTree* t = (TTree*)f->Get("EventTree");
    TTree* tgeo = (TTree*)f->Get("CellGeo");
    const ULong64_t nentries = t->GetEntries();

    // ------ Select branches. ------
    // We are copying EventTree -- but only certain branches.
    // The branches for clusters will be moved to a ClusterTree,
    // where we also add our images. These will be referenced by EventTree
    // in the "nCluster" and "clusterCount" branches (latter must be added).
    std::vector<TString> event_branches = {
    "runNumber", "eventNumber", "lumiBlock", "coreFlags",
    "mcEventNumber", "mcChannelNumber", "mcEventWeight",
    "nTruthPart",
    "G4PreCalo_n_EM", "G4PreCalo_E_EM",
    "G4PreCalo_n_Had", "G4PreCalo_E_Had",
    "truthPartPdgId",
    "truthPartStatus",
    "truthPartBarcode",
    "truthPartPt", "truthPartE", "truthPartMass", "truthPartEta", "truthPartPhi",
    "AntiKt4EMTopoJetsPt", "AntiKt4EMTopoJetsEta", "AntiKt4EMTopoJetsPhi", "AntiKt4EMTopoJetsE",
    "AntiKt4LCTopoJetsPt", "AntiKt4LCTopoJetsEta", "AntiKt4LCTopoJetsPhi", "AntiKt4LCTopoJetsE",
    "AntiKt4TruthJetsPt", "AntiKt4TruthJetsEta", "AntiKt4TruthJetsPhi", "AntiKt4TruthJetsE", "AntiKt4TruthJetsFlavor",
    "nCluster"};
    
    // cluster branches of type vector<float>
    std::vector<TString> cluster_branches_float = {
        "cluster_E", "cluster_E_LCCalib", "cluster_Pt", "cluster_Eta", "cluster_Phi",
        "cluster_ENG_CALIB_TOT", "cluster_ENG_CALIB_OUT_T", "cluster_ENG_CALIB_DEAD_TOT",
        "cluster_EM_PROBABILITY",
        "cluster_HAD_WEIGHT", "cluster_OOC_WEIGHT", "cluster_DM_WEIGHT",
        "cluster_CENTER_MAG", "cluster_FIRST_ENG_DENS", "cluster_CENTER_LAMBDA",
        "cluster_ISOLATION", "cluster_ENERGY_DigiHSTruth",
    };
    
    // cluster branches of type vector<int>
    std::vector<TString> cluster_branches_int = {
        "cluster_nCells"
    };
    
    // Reading was here!
    
    // ------ Prepare output file. ------
    // create the output file, and its TTrees (EventTree and ClusterTree)
    TFile* g = new TFile(outfile,"RECREATE");
    g->cd(); // in principle, this is redundant as we have just created g so we switch to it automatically as the current file
    
    // When we clone EventTree, we only turn on the branches we need to clone.
    t->SetBranchStatus("*",0);
    for (TString branch: event_branches) t->SetBranchStatus(branch,1);
    TTree* eventTree = t->CloneTree();
    
    // Now we turn on some extra branches, which will need to be read to fill ClusterTree.
    // TODO: Just turning on cluster_branches_float & cluster_branches_int causes empty cluster vars,
    // but turning on all branches does not. Why?
    t->SetBranchStatus("*",1);
    
    TTree* clusterTree = new TTree("ClusterTree","ClusterTree");
    
    // add the clusterCount branch to EventTree
    Long_t clusterCount;
    TBranch* br_clusterCount = eventTree->Branch("clusterCount",&clusterCount,"clusterCount/L");
    
    // add a bunch of branches to ClusterTree.
    // first we (neatly) handle the float branches
    std::map<TString,Float_t> cluster_f;
    for (TString branch: cluster_branches_float){
        cluster_f.insert(pair<TString, Float_t>(branch, 0.));
        TString branch_descriptor = TString(branch).Append("/F");
        clusterTree->Branch(branch, &cluster_f[branch],branch_descriptor);
    }

    // now handle int branches
    std::map<TString,Int_t> cluster_i;
    for (TString branch: cluster_branches_int){
        cluster_i.insert(pair<TString, Int_t>(branch, 0));
        TString branch_descriptor = TString(branch).Append("/I");
        clusterTree->Branch(branch, &cluster_i[branch],branch_descriptor);
    }
    
    // now add the image branches
    for(UInt_t i = 0; i < layers.size(); i++){
        TString descriptor = TString::Format("%s[%i][%i]/F",layers.at(i).Data(),len_eta.at(i),len_phi.at(i));
//         clusterTree->Branch(layers.at(i), &calo_images[(UInt_t)sampling_layers.at(i)], descriptor);
        clusterTree->Branch(layers.at(i), calo_images[(UInt_t)sampling_layers.at(i)].GetMatrixArray(), descriptor);

    }
    
    // ------ Setup for reading. ------
    
    // A) For reading EventTree
    TTreeReader mainReader(t);
    TTreeReaderArray<std::vector<ULong_t>> cluster_cell_id(mainReader,"cluster_cell_ID");
    TTreeReaderArray<std::vector<Float_t>> cluster_cell_E(mainReader,"cluster_cell_E");

    // for arrays of floats and ints, we use maps of TTreeReaderArray's for convenience later on.
    // see https://root-forum.cern.ch/t/storing-ttreereadervalues-in-std-map/21047/5
    // and https://root-forum.cern.ch/t/ttreereadervalue-in-std-map/37289
    // TODO: Does the below loop lead to a memory leak? Using "delete" line causes seg fault later on. This seems to work...
    std::map<TString,TTreeReaderArray<Float_t>*> readerArrays_f;
    for (TString branch: cluster_branches_float){
        TTreeReaderArray<Float_t>* ra = new TTreeReaderArray<Float_t>(mainReader,branch);
        readerArrays_f.insert(pair<TString, TTreeReaderArray<Float_t>*>(branch,ra));
//         delete ra;
    }
    
    std::map<TString,TTreeReaderArray<Int_t>*> readerArrays_i;
    for (TString branch: cluster_branches_int){
        TTreeReaderArray<Int_t>* ra = new TTreeReaderArray<Int_t>(mainReader,branch);
        readerArrays_i.insert(pair<TString, TTreeReaderArray<Int_t>*>(branch,ra));
//         delete ra;
    }
    
    // B) For reading CellGeo
    TTreeReader geoReader(tgeo);
    TTreeReaderArray<ULong_t> cell_geo_ID(geoReader,"cell_geo_ID");
    TTreeReaderArray<UShort_t> cell_geo_sampling(geoReader,"cell_geo_sampling");
    TTreeReaderArray<Float_t> cell_geo_eta(geoReader,"cell_geo_eta");
    TTreeReaderArray<Float_t> cell_geo_phi(geoReader,"cell_geo_phi");
    
    // extract cell_geo_ID vector from CellGeo tree
    geoReader.SetEntry(0); // since CellGeo is a single-entry tree (a bit of a funny structure for TTree...)
    std::vector<ULong_t> cgi;
    for(ULong_t entry: cell_geo_ID) cgi.push_back(entry);    
    
    clusterCount = 0;
    // ------ Event loop. ------
    for(ULong64_t i = 0; i < nentries; i++){
        mainReader.SetEntry(i);
//         std::cout << "Event: " << i << std::endl;
        Long_t ncluster = cluster_cell_id.GetSize();
        
        // ------ Cluster loop. ------
        for (Long_t j = 0; j < ncluster; j++){
//             std::cout << "\tCluster: " << j << std::endl;
            
            // flush all the images -- using "auto" here for simplicity
            for (auto entry: calo_images)  entry.second.Zero();
            
            // fill buffers for all scalar branches in ClusterTree
            for(TString branch: cluster_branches_float) cluster_f[branch] = readerArrays_f[branch]->At(j);
            for(TString branch: cluster_branches_int  ) cluster_i[branch] = readerArrays_i[branch]->At(j);
            
            ULong_t ncell = cluster_cell_id.At(j).size();
            
            // ------ Cell loop. ------
            for (ULong_t k = 0; k < ncell; k++){
//                 std::cout << "\t\tCell: " << k << std::endl;
                // i = event
                // j = clus
                // k = cell
                
                /* We want to "invert" the mapping from the cell_geo_ID branch:
                 * 0, val_0; 1, val_1; 2, val_2;
                 *           ->
                 * val_0, 0; val_1, 1; val_2, 2;
                 *
                 * (i.e. pass a value from that branch, return the entry number of that value)
                 * Since CellGeo is a single-entry tree with a vector, instead of a (nicer)
                 * multi-entry tree with a scalar, we cannot use TTreeIndex to do this lookup.
                 * Instead, we will do some searching within the vector cell_geo_ID using
                 * standard library functionality. 
                 */                
                ULong_t cci = cluster_cell_id.At(j).at(k); // cluster cell ID
                std::vector<ULong_t>::iterator it = std::find(cgi.begin(), cgi.end(), cci);
                Int_t index = std::distance(cgi.begin(), it);
                UShort_t cgs = cell_geo_sampling.At(index); // cell_geo_sampling value for this cell -- indicates the layer
                                
                if(std::find(sampling_layers.begin(), sampling_layers.end(), cgs) == sampling_layers.end()) continue;
                
                Int_t n_eta = calo_images[cgs].GetNrows();
                Int_t n_phi = calo_images[cgs].GetNcols();
                Float_t c_eta = readerArrays_f["cluster_Eta"]->At(j);
                Float_t c_phi = readerArrays_f["cluster_Phi"]->At(j);
                Float_t cg_eta = cell_geo_eta.At(index);
                Float_t cg_phi = cell_geo_phi.At(index);
                
                Int_t eta_bin = (Int_t)((cg_eta - c_eta - eta_min) * (Float_t)(n_eta) / eta_range);
                if(eta_bin < 0 || eta_bin >= n_eta) continue;
                
                Int_t phi_bin = (Int_t)((cg_phi - c_phi - phi_min) * (Float_t)(n_phi) / phi_range);
                if(phi_bin < 0 || phi_bin >= n_phi) continue;
                                
                // now add to the appropriate image
                Float_t cce = cluster_cell_E.At(j).at(k); // cluster cell energy
                Float_t c_E = readerArrays_f["cluster_E"]->At(j);
                calo_images[cgs](eta_bin,phi_bin) += cce/c_E; // note indexing for TMatrix in this way uses parentheses: https://root-forum.cern.ch/t/what-happened-to-tmatrixtrow/43726/4
            }
            
            // fill our ClusterTree
            clusterTree->Fill();
        }
        // fill our new EventTree (adding clusterCount)
        br_clusterCount->Fill();
        
        // now increase clusterCount
        clusterCount += ncluster;
    }
    
    f->Close();
    g->cd();
    eventTree->Write();
    clusterTree->Write();
    g->Close();
}

