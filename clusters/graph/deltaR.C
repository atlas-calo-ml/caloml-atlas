#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TString.h"
#include "TList.h"
#include "TObject.h"
#include "TObjArray.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"
#include <map>
#include <vector>
#include <algorithm>

void MakeTree(TString input_file, TString output_file){
    
    if(input_file.EqualTo(output_file)){
        return;
    }
    
    TFile* f = new TFile(input_file,"READ");
    TTree* ct = (TTree*)f->Get("ClusterTree");
    TTree* et = (TTree*)f->Get("EventTree");
    Long64_t nevents = et->GetEntries();
    
    // Set up readers.
    TTreeReader ct_reader(ct);
    TTreeReader et_reader(et);
    
    TTreeReaderValue<Int_t> nCluster(et_reader,"nCluster");
    TTreeReaderValue<Long64_t> clusterCount(et_reader,"clusterCount");
    TTreeReaderArray<Float_t> truthPartPt(et_reader,"truthPartPt");
    TTreeReaderArray<Float_t> truthPartEta(et_reader,"truthPartEta");
    TTreeReaderArray<Float_t> truthPartPhi(et_reader,"truthPartPhi");
    TTreeReaderArray<Float_t> truthPartE(et_reader,"truthPartE");
    TTreeReaderArray<Int_t> truthPartStatus(et_reader,"truthPartStatus");
    TTreeReaderArray<Int_t> truthPartPdgId(et_reader,"truthPartPdgId");
    
    TTreeReaderValue<Float_t> cluster_E(ct_reader,"cluster_E");
    TTreeReaderValue<Float_t> cluster_Eta(ct_reader,"cluster_Eta");
    TTreeReaderValue<Float_t> cluster_Phi(ct_reader,"cluster_Phi");
    TTreeReaderValue<Float_t> cluster_ENG_CALIB_TOT(ct_reader,"cluster_ENG_CALIB_TOT");

    // Set up output TTree.
    TFile* g = new TFile(output_file,"RECREATE");
    TTree* t = new TTree("dr_info","dr_info");
    
    // Writing buffers
    Float_t dR;
    Float_t clus_E;
    Float_t clus_ENG_CALIB_TOT;
    Float_t clus_Eta;
    Float_t clus_Phi;
    
    Float_t truth_E;
    Float_t truth_Eta;
    Float_t truth_Phi;
    Int_t truth_Pdg;
    
    t->Branch("dR",&dR, "dR/F");
    t->Branch("cluster_E",&clus_E, "cluster_E/F");
    t->Branch("cluster_Eta",&clus_Eta, "cluster_Eta/F");
    t->Branch("cluster_Phi",&clus_Phi, "cluster_Phi/F");
    t->Branch("cluster_ENG_CALIB_TOT",&clus_ENG_CALIB_TOT, "cluster_ENG_CALIB_TOT/F");

    t->Branch("truth_E",&truth_E, "truth_E/F");
    t->Branch("truth_Eta",&truth_Eta, "truth_Eta/F");
    t->Branch("truth_Phi",&truth_Phi, "truth_Phi/F");
    t->Branch("truth_PdgId",&truth_Pdg, "truth_PdgId/I");

    // vectors for our deltaR calculations
    ROOT::Math::PtEtaPhiEVector* v1 = new ROOT::Math::PtEtaPhiEVector();
    ROOT::Math::PtEtaPhiEVector* v2 = new ROOT::Math::PtEtaPhiEVector();
    
    std::vector<Float_t> dR_vals;
    std::vector<Int_t> k_vals;
    
    for(Long64_t i = 0; i < nevents; i++){
        et_reader.SetEntry(i);
        
        UInt_t n_truth = truthPartEta.GetSize();
        for(Long64_t j = 0; j < *nCluster; j++){
            ct_reader.SetEntry(*clusterCount + j);
            
            clus_E = *cluster_E;
            clus_ENG_CALIB_TOT = *cluster_ENG_CALIB_TOT;
            clus_Eta = *cluster_Eta;
            clus_Phi = *cluster_Phi;

            dR_vals.clear();
            k_vals.clear();

            for(Int_t k = 0; k < n_truth; k++){
                if(truthPartStatus.At(k) < 0) continue;
                v1->SetCoordinates(0.,clus_Eta, clus_Phi, 0.);
                v2->SetCoordinates(0., truthPartEta.At(k), truthPartPhi.At(k), 0.);
                dR_vals.push_back(ROOT::Math::VectorUtil::DeltaR(*v1,*v2));
                k_vals.push_back(k); // keeping track of index since we have the skip condition above
            }
            
            // Get the index of the smallest dR.
            Int_t min_index = std::min_element(dR_vals.begin(),dR_vals.end()) - dR_vals.begin();
            dR = dR_vals.at(min_index);
            Int_t truth_index = k_vals.at(min_index);
            truth_E   = truthPartE.At(truth_index);
            truth_Eta = truthPartEta.At(truth_index);
            truth_Phi = truthPartPhi.At(truth_index);
            truth_Pdg = truthPartPdgId.At(truth_index);
                        
            t->Fill();
        }
    }
    g->cd();
    t->Write();
    g->Close();
    f->Close();
    return;
}

void deltaR(TString input_file, TString output_file){
    MakeTree(input_file, output_file);
}