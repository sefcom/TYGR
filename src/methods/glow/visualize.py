from typing import *

from .common import GlowInput, GlowOutput

def visualize(base_file_name: str, datapoint: Tuple[GlowInput, GlowOutput]):
  (glow_input, glow_output) = datapoint

  mark_nodes = []

  with open(f"{base_file_name}.txt", "w") as f:
    f.write(f"Directory: {glow_input.directory}\n")
    f.write(f"File Name: {glow_input.file_name}\n")
    f.write(f"Function Name: {glow_input.function_name}\n")
    f.write(f"Low PC: {glow_input.low_high_pc[0]}\n")
    f.write(f"High PC: {glow_input.low_high_pc[1]}\n")
    f.write(f"Vars:\n")

    for (i, (glow_var, var_type)) in enumerate(zip(glow_input.vars, glow_output.types)):
      f.write(f"|- Var {i}:\n")
      f.write(f"|  |- Var Name: {glow_var.name}\n")
      f.write(f"|  |- Offset: {glow_var.locs}\n")
      f.write(f"|  |- Nodes: {glow_var.nodes}\n")
      f.write(f"|  |- Type: {str(var_type)}\n")
      mark_nodes += glow_var.nodes

  glow_input.ast_graph.save_dot(f"{base_file_name}.dot", mark_nodes=set(mark_nodes))
