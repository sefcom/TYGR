from src.methods.glow.common import GlowInput, GlowOutput
import pickle
import os
from typing import *
from src.analysis.types.types import *
import numpy as np
from multiprocessing import Process
import argparse
import sys
sys.setrecursionlimit(5000)

def filter_ill_formed(list_samples):
    # Stats
    source_num_functions = len(list_samples)
    source_num_vars = 0
    preproc_num_vars = 0

    result = []
    for sample in list_samples:
      (glow_input, glow_output) = sample

      # Stats
      source_num_vars += len(glow_input.vars)

      if len(glow_input.vars) == 0:
        continue
      new_vars = [v for v in glow_input.vars if len(v.nodes) > 0]
      new_types = [v for (i, v) in enumerate(glow_output.types) if len(glow_input.vars[i].nodes) > 0]

      # Stats
      preproc_num_vars += len(new_vars)

      if len(new_vars) == 0:
        continue
      glow_input.vars = new_vars
      glow_output.types = new_types
      result.append(sample)

    # Stats
    # preproc_num_functions = len(result)
    # print("Source #functions:", source_num_functions)
    # print("Well Formed #functions:", preproc_num_functions)
    # print("Source #vars:", source_num_vars)
    # print("Well Formed #vars:", preproc_num_vars)

    return result

def transform_dataset_get_struct_member(pkl_file, out_file):
    
    dataset = pickle.load(open(pkl_file, "rb"))
    
    new_dataset = list(get_struct_member(dataset))
    if new_dataset == None:
        print("dataset none")
        return
    
    new_dataset = filter_ill_formed(new_dataset)
    with open(out_file, "wb") as f:
        pickle.dump(new_dataset, f)

def transform_dataset_get_struct(pkl_file, out_file):     
    
    dataset = pickle.load(open(pkl_file, "rb"))
    new_dataset = list(get_struct(dataset))
    
    if new_dataset == None:
        print("dataset none")
        return
    
    new_dataset = filter_ill_formed(new_dataset)
    with open(out_file, "wb") as f:
        pickle.dump(new_dataset, f)
            
def get_struct_member(dataset: List[Tuple[GlowInput, GlowOutput]]):
    for (i, o) in dataset:
        tmp_vars = []
        tmp_types = []
        for index in range(len(i.vars)):
            if "struct_" in str(o.types[index]):
                tmp_vars.append(i.vars[index])
                # only elem
                tmp_types.append(o.types[index].elem)
          
        i.vars = tmp_vars
        o.types = tmp_types
        yield (i, o)
  
def get_struct(dataset: List[Tuple[GlowInput, GlowOutput]]):
    for (i, o) in dataset:
        for index in range(len(i.vars)):
            if "struct_" in str(o.types[index]):         
                o.types[index] = StructType()
        yield (i, o)     


parser = argparse.ArgumentParser(description='Model Pkl Generation')
parser.add_argument("merged_file", type=str)
parser.add_argument("model_base_pkl", type=str)
parser.add_argument("model_struct_pkl", type=str)

args = parser.parse_args()

merged_file = args.merged_file
model_base_pkl = args.model_base_pkl
model_struct_pkl = args.model_struct_pkl


process1 = Process(target=transform_dataset_get_struct, args=(merged_file, model_base_pkl,))
process2 = Process(target=transform_dataset_get_struct_member, args=(merged_file, model_struct_pkl, ))

# Start the processes
process1.start()
process2.start()

# Wait for both processes to finish
process1.join()
process2.join()

print("Both processes have completed")     