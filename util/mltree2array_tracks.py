#! /usr/bin/env python
import numpy as np
import sys
import ROOT

def load_tree(files, tree, branches, nmax = -1, selection=''):
  """ Load specified branches from the input TTrees into memory"""

  ROOT.PyConfig.IgnoreCommandLineOptions = True
  ROOT.gROOT.SetBatch(True)
  chain = ROOT.TChain(tree)
  for f in files: chain.Add(f)

  from root_numpy import tree2array
  return tree2array(chain, branches = branches, selection = selection, start = 0, stop = nmax)

def merge_clusters_to_tracks(clusters, tracks, branches):
  merged_array = []
  track_len = len(tracks[0])
  print(len(clusters))
  for cluster in clusters:
      run_number = cluster[0]
      event_number = cluster[1]
      event_tracks = [track for track in tracks if track[0] == run_number and track[1] == event_number][0]

      cluster_eta = cluster[10]
      cluster_phi = cluster[11]
      cluster_tlv = ROOT.TLorentzVector()
      cluster_tlv.SetPtEtaPhiE(1, cluster_eta, cluster_phi, 1)

      num_of_tracks = len(event_tracks[2])
      # If no tracks fill with -10, chosen as impossible value for real track but not too far like -999
      if num_of_tracks == 0:
          merge_track = (-10,) * (track_len - 2)
      # Else create tuple with info of track closest in dR
      else:
          min_dr = float('inf')
          min_i = 0
          for i in range(num_of_tracks):
              track_eta = event_tracks[3][i]
              track_phi = event_tracks[4][i]
              track_tlv = ROOT.TLorentzVector()
              track_tlv.SetPtEtaPhiE(1, cluster_eta, cluster_phi, 1)
              dr = cluster_tlv.DeltaR(track_tlv)
              if dr < min_dr:
                  min_dr = dr
                  min_i = i

          merge_track = [branch[min_i] for branch in event_tracks.tolist()[2:]]
          # Replace default MLTree value of -1000000000 with -10
          for i in range(len(merge_track)):
              if merge_track[i] < -10000:
                  merge_track[i] = -10 
          merge_track = tuple(merge_track)

      merge_cluster = cluster.tolist()[2:]
      merged_info = merge_cluster + merge_track
      merged_array.append(merged_info)
  return merged_array

def preprocess(clusters, branches, flatten = False, label = 0):
  """ Pre-processing of the CaloML image dataset """

  ncl = len(clusters)
  nbr = len(branches)

  # one image for each layer of the calorimeter
  data = {
     # EM barrel
    'EMB1': np.zeros((ncl, 128, 4)),
    'EMB2': np.zeros((ncl, 16, 16)),
    'EMB3': np.zeros((ncl, 8, 16)),
    # TileCal barrel
    'TileBar0': np.zeros((ncl, 4, 4)),
    'TileBar1': np.zeros((ncl, 4, 4)),
    'TileBar2': np.zeros((ncl, 2, 4))
  }

  # supplemental info about clusters and cells (clusE, clusPt, nCells,...)
  for br in branches[6:]:
    data[br] = np.zeros(ncl)

  # fill the image arrays and the supplemental info
  for i in range(ncl):
    for j in range(nbr):
      data[branches[j]][i] = clusters[i][j]
      if flatten: data[branches[j]][i] = data[branches[j]][i].flatten()

  # add a vector of labels
  data['label'] = np.full((ncl, 1), label)

  return data

def export(data, output, compress):
  """ Export data to file """

  if compress:
    np.savez_compressed(output, **data)
  else:
    np.save(output, **data)

if __name__ == "__main__":

  #default_cluster_branches = ['runNumber', 'eventNumber', 'EMB1', 'EMB2', 'EMB3', 'TileBar0', 'TileBar1', 'TileBar2', 'clusterE', 'clusterPt', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_sumCellE', 'cluster_emProb']
  default_cluster_branches = ['runNumber', 'eventNumber', 'EMB1', 'EMB2', 'EMB3', 'TileBar0', 'TileBar1', 'TileBar2', 'clusterE', 'clusterPt', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_sumCellE']
  default_event_branches = ['runNumber', 'eventNumber', 'trackPt', 'trackEta', 'trackPhi',
                                                                    'trackEta_PreSamplerB', 'trackPhi_PreSamplerB',
                                                                    'trackEta_PreSamplerE', 'trackPhi_PreSamplerE',
                                                                    'trackEta_EMB1', 'trackPhi_EMB1',
                                                                    'trackEta_EMB2', 'trackPhi_EMB2',
                                                                    'trackEta_EMB3', 'trackPhi_EMB3',
                                                                    'trackEta_EME1', 'trackPhi_EME1',
                                                                    'trackEta_EME2', 'trackPhi_EME2',
                                                                    'trackEta_EME3', 'trackPhi_EME3',
                                                                    'trackEta_HEC0', 'trackPhi_HEC0',
                                                                    'trackEta_HEC1', 'trackPhi_HEC1',
                                                                    'trackEta_HEC2', 'trackPhi_HEC2',
                                                                    'trackEta_HEC3', 'trackPhi_HEC3',
                                                                    'trackEta_TileBar0', 'trackPhi_TileBar0',
                                                                    'trackEta_TileBar1', 'trackPhi_TileBar1',
                                                                    'trackEta_TileBar2', 'trackPhi_TileBar2',
                                                                    'trackEta_TileGap1', 'trackPhi_TileGap1',
                                                                    'trackEta_TileGap2', 'trackPhi_TileGap2',
                                                                    'trackEta_TileGap3', 'trackPhi_TileGap3',
                                                                    'trackEta_TileExt0', 'trackPhi_TileExt0',
                                                                    'trackEta_TileExt1', 'trackPhi_TileExt1',
                                                                    'trackEta_TileExt2', 'trackPhi_TileExt2',
                                                                    'trackNumberOfPixelHits',
                                                                    'trackNumberOfSCTHits',
                                                                    'trackNumberOfPixelDeadSensors',
                                                                    'trackNumberOfSCTDeadSensors',
                                                                    'trackNumberOfPixelSharedHits',
                                                                    'trackNumberOfSCTSharedHits',
                                                                    'trackNumberOfPixelHoles',
                                                                    'trackNumberOfSCTHoles',
                                                                    'trackNumberOfInnermostPixelLayerHits',
                                                                    'trackNumberOfNextToInnermostPixelLayerHits',
                                                                    'trackExpectInnermostPixelLayerHit',
                                                                    'trackExpectNextToInnermostPixelLayerHit',
                                                                    'trackNumberOfTRTHits',
                                                                    'trackNumberOfTRTOutliers',
                                                                    'trackChiSquared',
                                                                    'trackNumberDOF',
                                                                    'trackD0',
                                                                    'trackZ0',
                                                                    ]

  import argparse
  parser = argparse.ArgumentParser(add_help=True, description='Convert root image arrays from the MLTree package to numpy arrays.')
  # Arguments relevant to the cluster tree
  parser.add_argument('--cluster_files', type=str, nargs='+', metavar='<file.root>', help='ROOT files containing the outputs from the MLTree package run in cluster mode.')
  parser.add_argument('--cluster_label', '-cl', required=True, type=int, help='Label for images in input array from cluster file')
  parser.add_argument('--cluster_tree', required=False, type=str, help='Name of input cluster TTree.', default='ClusterTree')
  parser.add_argument('--cluster_branches', required=False, type=str, nargs='+', help='List of branch names to import from cluster file', default = default_cluster_branches)
  #Arguments relevant to the event tree containing tracks
  parser.add_argument('--event_files', type=str, nargs='+', metavar='<file.root>', help='ROOT files containing the outputs from the MLTree package run in event mode.')
  parser.add_argument('--event_label', '-el', required=True, type=int, help='Label for images in input array from event file')
  parser.add_argument('--event_tree', required=False, type=str, help='Name of input event TTree.', default='EventTree')
  parser.add_argument('--event_branches', required=False, type=str, nargs='+', help='List of branch names to import from event file', default = default_event_branches)
  #Arguments relevant generally
  parser.add_argument('--output', '-o', required=False, type=str, help='Output file to store the images', default='images')
  parser.add_argument('--compress', '-c', required=False, action='store_true', help='Compress output arrays.', default=False)
  parser.add_argument('--nclusters', '-n', required=False, type=int, help='Number of clusters to process', default=-1)
  parser.add_argument('--flatten', required=False, action='store_true', help='Flatten output arrays', default=False)
  args = parser.parse_args()

  branches = default_cluster_branches[2:] + default_event_branches[2:]

  print("loading data from tree...")
  clusters = load_tree(args.cluster_files, args.cluster_tree, args.cluster_branches, args.nclusters, 'clusterE > 100')
  events = load_tree(args.event_files, args.event_tree, args.event_branches)
  print("merging clusters to tracks...")
  merged = merge_clusters_to_tracks(clusters, events, branches)
  print("pre-processing data...")
  data = preprocess(merged, branches, args.flatten, 0)
  print("saving data...")
  export(data, args.output, args.compress)
  print("\nall done!")
