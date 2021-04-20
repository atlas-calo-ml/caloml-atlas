import sys, os, glob, argparse, time, datetime
import subprocess as sub
import ROOT as rt

path_prefix = os.getcwd() + '/../../'
if(path_prefix not in sys.path): sys.path.append(path_prefix)
from util import qol_util as qu

def main(args):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file directory',required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file directory',required=False)
    args = vars(parser.parse_args())
    
    input_dir = args['input']
    
    if('output' in args.keys()): output_dir = args['output']
    else: output_dir = os.getcwd()
        
    input_files = glob.glob(input_dir + '/**/*.root',recursive=True)
    output_files = [x.replace(input_dir,output_dir) for x in input_files]
    
    n = len(input_files)
    print('Converting {} files.'.format(n))
    
    prefix = 'Converting files:'
    suffix = 'Complete'
    qu.printProgressBarColor (0, n, prefix=prefix, suffix=suffix, length=50)
    start_time = time.time()
    for i in range(n):
        
        output_subdir = '/'.join(output_files[i].split('/')[:-1])
        try: os.makedirs(output_subdir)
        except: pass
        
        command = 'root -q -l -b -x \'g2i.C+("{}", "{}")\''.format(input_files[i], output_files[i])
        sub.check_call(command, shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL)
        qu.printProgressBarColor (i+1, n, prefix=prefix, suffix=suffix, length=50)
    
    end_time = time.time()
    delta_time = int(end_time - start_time)
    print('Time elapse: {}.'.format(str(datetime.timedelta(seconds=delta_time))))
    
if __name__ == '__main__':
    main(sys.argv)