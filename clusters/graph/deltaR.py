import sys, os, glob, argparse, pathlib
import subprocess as sub

path_prefix = os.getcwd() + '/../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import qol_util as qu

def main(args):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file directory',required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file directory',required=False)
    parser.add_argument('-f', '--force', type=bool, help='Overwrite existing output',required=False,default=False)
    args = vars(parser.parse_args())
    
    if(args['output'] != None): output_dir = args['output']
    else: output_dir = os.getcwd()
            
    # Get input files.
    input_dir = args['input']
    input_files = glob.glob(input_dir + '/**/*.root',recursive=True)
    output_files = [x.replace(input_dir, output_dir) for x in input_files]
    
    n = len(output_files)
    prefix = 'Computing min dR info'
    suffix = 'Complete'
    print('\n')
    qu.printProgressBarColor (0, n, prefix=prefix, suffix=suffix, length=50)
    
    for i, ofile in enumerate(output_files):
        
        output_subdir = '/'.join(ofile.split('/')[:-1])
        try: os.makedirs(output_subdir)
        except: pass
        if((not args['force']) and pathlib.Path(ofile).exists()): 
            qu.printProgressBarColor (i+1, n, prefix=prefix, suffix=suffix, length=50)
            continue
        
        command = 'root -l -b -q -x \'deltaR.C+(\"{}\",\"{}\")\''.format(input_files[i], ofile)
        sub.check_call(command,shell=True,stdout=sub.DEVNULL)
        qu.printProgressBarColor (i+1, n, prefix=prefix, suffix=suffix, length=50)
    return

if __name__ == '__main__':
    main(sys.argv)