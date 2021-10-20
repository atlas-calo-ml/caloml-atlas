import sys, os, glob, argparse, time, datetime, pathlib
import subprocess as sub
import uproot as ur

path_prefix = os.getcwd() + '/../../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import qol_util as qu

def main(args):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file directory',required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file directory',required=False)
    parser.add_argument('-f', '--force', type=bool, help='Overwrite any existing output',default=False)
    args = vars(parser.parse_args())
    
    input_dir = args['input']
    
    if('output' in args.keys()): output_dir = args['output']
    else: output_dir = os.getcwd()
        
    force = args['force']
        
    input_files = glob.glob(input_dir + '/**/*.root',recursive=True)
    output_files = [x.replace(input_dir,output_dir) for x in input_files]
    
    n = len(input_files)
    print('Converting {} files.'.format(n))
    
    prefix = 'Converting files:'
    suffix = 'Complete'
    qu.printProgressBarColor (0, n, prefix=prefix, suffix=suffix, length=50)
    
    check_keys = ['EventTree','ClusterTree']
    
    nrun = 0
    
    start_time = time.time()
    for i in range(n):
        
        output_subdir = '/'.join(output_files[i].split('/')[:-1])
        try: os.makedirs(output_subdir)
        except: pass
        
        # Optionally check that the output file already exists
        # (and isn't broken), skip if it already exists.
        run_single = True
        if (not force):
            if(pathlib.Path(output_files[i]).exists()):
                with ur.open(output_files[i]) as f:
                    fkeys = f.keys()
                    for key in check_keys:
                        if(key not in fkeys):
                            run_single = False
                            break
                        else:
                            t = f[key]
                            nt = t.num_entries
                            if(nt <= 0):
                                run_single = False
                                break
                            
        command = 'root -q -l -b -x \'g2i.C+("{}", "{}")\''.format(input_files[i], output_files[i])
        if(run_single): 
            nrun += 1
            sub.check_call(command, shell=True, stdout=sub.DEVNULL)
        qu.printProgressBarColor (i+1, n, prefix=prefix, suffix=suffix, length=50)
    
    end_time = time.time()
    delta_time = int(end_time - start_time)
    print('Time elapse: {}.'.format(str(datetime.timedelta(seconds=delta_time))))
    
    if(n-nrun > 0): 
        print('Number of files converted: {}\t(Found {} files already converted. Use \'-f 1\' option to overwrite these.)'.format(nrun, n-nrun))
    else: 
        print('Number of files converted: {}'.format(nrun))

if __name__ == '__main__':
    main(sys.argv)