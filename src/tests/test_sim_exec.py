import sys
import os.path

# Parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from argparse import ArgumentParser

import angr
from analysis.angr.sim_exec import *
from analysis.angr.ast_graph import *

def parser():
  parser = argparse.ArgumentParser(description="Test the symbolic execution")
  parser.add_argument("-i", "--input_file", type=str, required=True)
  parser.add_argument("-a", "--init_addr", type=str)
  parser.add_argument("-v", "--verbose", action="store_true")
  return parser

args = parser().parse_args()

# Initialize angr stuff
load_opts = {"main_opts" : {"base_addr" : 0x0}, "auto_load_libs" : False}
proj = angr.Project(args.input_file, load_options=load_opts)
arch = proj.arch

cfg = proj.analyses.CFGFast()
# cfg = proj.analyses.CFGFast(data_references=True, force_complete_scan=False)
# cfg = proj.analyses.CFGEmulated()

if args.init_addr:
  init_addr = int(args.init_addr, 16)
else:
  init_addr = proj.entry

# Initialize the strategy we use
print("about to initialize and execute strategy")
# strat = SimpleDominatorStrategy(proj, {"cfg" : cfg})

strat_config = {
  "cfg" : cfg,
  "verbose" : args.verbose
}

strat = LessSimpleDominatorStrategy(proj, config=strat_config)
res = strat.sim_exec_function(init_addr)
statefs = []
for (_, _, sf) in res.tups:
  statefs.append(sf)

## Specific code for one stuff

def print_kvs(kvs):
  for k, v in kvs.items():
    print(f"{k} \t {v}")

def print_kvalues(state, kvs):
  for k, (a, v) in kvs.items():
    print(f"{hex(state.solver.eval(k))} \t {hex(state.solver.eval(v))}")

regs0, mems0 = get_state_writes(statefs[0])
regs1, mems1 = get_state_writes(statefs[1])

# Get the variables

b0_mem, (b0_addr, b0_expr) = list(mems0.items())[3]
p0_mem, (p0_addr, p0_expr) = list(mems0.items())[4]
r0_mem, (r0_addr, r0_expr) = list(mems0.items())[5]

b1_mem, (b1_addr, b1_expr) = list(mems1.items())[3]
p1_mem, (p1_addr, p1_expr) = list(mems1.items())[4]
r1_mem, (r1_addr, r1_expr) = list(mems1.items())[5]

# The augmented annotations

b0_aug = statefs[0].solver.BVS("", 32).annotate(MemoryWriteAnnotation(b0_addr, b0_mem, b0_expr))
p0_aug = statefs[0].solver.BVS("", 64).annotate(MemoryWriteAnnotation(p0_addr, p0_mem, p0_expr))
r0_aug = statefs[0].solver.BVS("", 32).annotate(MemoryWriteAnnotation(r0_addr, r0_mem, r0_expr))

b1_aug = statefs[1].solver.BVS("", 32).annotate(MemoryWriteAnnotation(b1_addr, b1_mem, b1_expr))
p1_aug = statefs[1].solver.BVS("", 64).annotate(MemoryWriteAnnotation(p1_addr, p1_mem, p1_expr))
r1_aug = statefs[1].solver.BVS("", 32).annotate(MemoryWriteAnnotation(r1_addr, r1_mem, r1_expr))

# Add constraints
statefs[0].solver.add(b0_aug == b0_expr)
statefs[0].solver.add(p0_aug == p0_expr)
statefs[0].solver.add(r0_aug == r0_expr)

statefs[1].solver.add(b1_aug == b1_expr)
statefs[1].solver.add(p1_aug == p1_expr)
statefs[1].solver.add(r1_aug == r1_expr)
print("hello I am here !!!")
# The graphs
b0_graph = AstGraph()
b0_graph.process_ast(statefs[0], b0_expr)

p0_graph = AstGraph()
p0_graph.process_ast(statefs[0], p0_expr)

r0_graph = AstGraph()
r0_graph.process_ast(statefs[0], r0_expr)

b1_graph = AstGraph()
b1_graph.process_ast(statefs[1], b1_expr)

p1_graph = AstGraph()
p1_graph.process_ast(statefs[1], p1_expr)

r1_graph = AstGraph()
r1_graph.process_ast(statefs[1], r1_expr)

big_graph = AstGraph()
b0_node = big_graph.process_ast(statefs[0], b0_expr)
p0_node = big_graph.process_ast(statefs[0], p0_expr)
b1_node = big_graph.process_ast(statefs[1], b1_expr)
p1_node = big_graph.process_ast(statefs[1], p1_expr)

print(f"b0: {b0_node}")
print(f"p0: {p0_node}")
print(f"b1: {b1_node}")
print(f"p1: {p1_node}")

b0_graph.save_dot("/home/taro/foo/test/b-true.dot")
p0_graph.save_dot("/home/taro/foo/test/p-true.dot")
b1_graph.save_dot("/home/taro/foo/test/b-false.dot")
p1_graph.save_dot("/home/taro/foo/test/p-false.dot")
big_graph.save_dot("/home/taro/foo/test/big.dot")



