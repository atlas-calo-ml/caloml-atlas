import pprint
#qkeras imports
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings

# get information on energy consumption of a network
def GetEnergy(model, verbose=False):
    # energy estimation
    reference_internal    = 'fp32'
    reference_accumulator = 'fp32'
    proc = 'horowitz'

    q = run_qtools.QTools(
          model,
          process=proc,
          source_quantizers=[quantized_bits(8, 0, 1)],
          is_inference=False,
          weights_path=None,
          keras_quantizer=reference_internal,
          keras_accumulator=reference_accumulator,
          # whether calculate baseline energy
          for_reference=True)
    
    energy_dict = q.pe(
    weights_on_memory="sram",
    activations_on_memory="sram",
    min_sram_size=8*16*1024*1024, # minimum sram size in number of bits. Let's assume a 16MB SRAM.
    rd_wr_on_io=False) # assuming data alreadu in SRAM

    energy_profile = q.extract_energy_profile(qtools_settings.cfg.include_energy, energy_dict)
    total_energy = q.extract_energy_sum(qtools_settings.cfg.include_energy, energy_dict)
    if(verbose): 
        pprint.pprint(energy_profile)
        print()
    print("Total energy: {:.2f} uJ".format(total_energy / 1000000.0))
    return

def GetCostReduction(filename):
    if('.tf' in filename):
        filename = filename.replace('.tf','.log')
    
    reduction = 0
    idx = 0
    
    with open(filename,'r') as f:
        lines = f.readlines()
        
    for i,line in enumerate(lines):
        if('Total Cost Reduction:') in line:
            idx = i +1
            break
            
    reduction = lines[idx].split('(')[-1].replace(')','').strip()
    return reduction