import sys, os, glob
import argparse as ap
import numpy as np
import subprocess as sub

def MatchJets(match_string, match_settings, input_files, output_file, executable_suffix=''):
    
    util_dir = os.path.dirname(os.path.realpath(__file__))
    setting_string = ','.join([str(match_settings[k]) for k in match_settings.keys()])
    setting_string = setting_string.replace('True','1').replace('False','0')
    
    n_files = len(input_files)
    output_files = [x.replace('.root','_out.root') for x in input_files]
    
    if(executable_suffix != '' and executable_suffix[-1] != '/'): executable_suffix +='/'

    for i in range(n_files):
        command = 'root -q -l -b -x \'{}JetMatching.C+("{}", "{}", "{}", "{}")\''.format(executable_suffix, input_files[i], output_files[i], match_string, setting_string)
        sub.check_call(command,shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL) # TODO: Try to switch to shell=False version

    if(n_files > 1):  
        # Combine our output files, and delete the temporary files.
        command = ['hadd', output_file] + output_files
        sub.check_call(command,shell=False, stdout=sub.DEVNULL)
        command = ['rm'] + output_files
        sub.check_call(command,shell=False)
        
    else:
        command = ['mv', output_files[0], output_file]
        sub.check_call(command,shell=False, stdout=sub.DEVNULL)
        
    return

# def main(args):
    
#     # Some preset matching stuff.
    
#     #TODO: For a 2-part match a->b->c, we currently have to call matches "b->c;a->b;b->c".
#     # The repeat of b->c at the end deals with the case where some species-b jets are thrown out during "a->b",
#     # in which case we need the 2nd "b->c" match to toss out the corresponding species-c jets and make sure that
#     # all three species (a,b,c) have the exact same number of jets. (i.e. ensure no partial matches of any kind, only full matches).
    
#     # TL;DR If you are matching some sequence of jets and you want to make sure that you are only keeping complete matches in that sequence,
#     # your list of matches and match_settings below must be palindromes (at the level of entries).
    
#     # NOTE: I do not think that this workaround is perfect. I suspect there are cases (rare?) where this yields a different result than I claim.
#     # In the long-run, I will need to modify JetMatching.C accordingly to allow the user to pass along "history" of previous matches.
    
#     matches = [
#         'AntiKt4lctopoCaloCalJets->AntiKt4TruthJets'
# #         'AntiKt4emtopoCalo422VorSKJets->AntiKt4lctopoCaloCalJets'
#     ]
#     match_settings = [
#         {
#             'pt_min':7.0e3,
#             'eta_max':4.5,
#             'requirePileupCheck':True,
#             'requireIsoReco':True,
#             'requireIsoTruth':True,
#             'dr':0.3,
#             'truth_iso_dr':0.3,
#             'reco_iso_dr':0.3
#         }        
# #         {
# #             'pt_min':7.0e3,
# #             'eta_max':4.5,
# #             'requirePileupCheck':False,
# #             'requireIsoReco':True,
# #             'requireIsoTruth':True,
# #             'dr':0.3,
# #             'truth_iso_dr':0.3,
# #             'reco_iso_dr':0.3
# #         }
#     ]
    
#     match_string = ';'.join(matches)
#     setting_substrings = [','.join([str(x[k]) for k in x.keys()]) for x in match_settings]
#     setting_substrings = [x.replace('True','1').replace('False','0') for x in setting_substrings]
#     setting_string = ';'.join(setting_substrings)
    
#     parser = ap.ArgumentParser()
#     parser.add_argument('-i', '--input', type=str, help='Input directory.',required=True)
#     parser.add_argument('-o', '--outdir', type=str, help='Output directory.',required=True)
#     parser.add_argument('-c', '--concat', type=bool, help='Do concatenation of output.',default=False)
#     parser.add_argument('-O', '--outname', type=str, help='Output file name (if concatenating).',default='out.root')
#     args = vars(parser.parse_args())
    
#     input_files = glob.glob(args['input'] + '/**/*.root*',recursive=True)
    
#     try: os.makedirs(args['outdir'])
#     except: pass
    
#     n_files = len(input_files)
#     nz = int(np.ceil(np.log10(n_files)))
#     output_files = [args['outdir'] + '/out_{}.root'.format(str(x).zfill(nz)) for x in range(len(input_files))]
#     output_files = [x.replace('//','/') for x in output_files]
    
#     for i in range(n_files):
#         command = 'root -q -l -b -x \'JetMatching.C+("{}", "{}", "{}", "{}")\''.format(input_files[i], output_files[i], match_string, setting_string)
#         sub.check_call(command,shell=True)
        
#     if(args['concat']):
#         output_file = args['outdir'] + '/' + args['outname']
        
#         if(n_files > 1):
#             command = 'hadd {} {}/out_*.root'.format(output_file, args['outdir'])
#             command = command.replace('//','/')
#             sub.check_call(command,shell=True)
#             command = 'rm {}/out_*.root'.format(args['outdir'])
#             sub.check_call(command,shell=True)
#         else:
#             command = 'mv {}/out_0.root {}'.format(args['outdir'],output_file)
#             sub.check_call(command,shell=True)
#     return
    
# if __name__ == '__main__':
#     main(sys.argv)
