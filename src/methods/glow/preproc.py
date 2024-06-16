from typing import *
import torch
from torch import Tensor

from claripy.operations import expression_operations, backend_operations_all, backend_fp_operations, backend_strings_operations
from archinfo.arch_arm import ArchARM
from archinfo.arch_mips32 import ArchMIPS32
from archinfo.arch_amd64 import ArchAMD64
from archinfo.arch_x86 import ArchX86
from archinfo.arch_aarch64 import ArchAArch64

from ...analysis.types.type_set import get_type_set
from ...analysis.angr.ast_graph import NodeLabel
from ...analysis.angr.edge import REDUCED_OPS, REDUCE_OP_MAP
from .common import GlowInput, PreprocGlowInput, GlowOutput

SP_RANGE = 512
NUM_FEATURES = 18

class Config:
  def __init__(
    self,
    type_set: str = "rstd",
    bitsize: int = 32,
    use_bitvector: bool = False,
    use_raw_edges: bool = False,
    flatten_vars: bool = True,
    use_arch: bool = True,
  ):
    self.use_bitvector = use_bitvector
    self.flatten_vars = flatten_vars
    self.use_arch = use_arch

    # Type embedding translation
    self.type_set = get_type_set(type_set)

    # Node embedding translation
    self.bitsize = bitsize
    self.arch_fields = ["x64", "x86", "arm32", "arm64", "mips"]
    self.bitsize_fields = [1, 8, 16, 32, 64, 128, 256]
    self.unaligned_bitsize_index = len(self.bitsize_fields)
    self.bitsize_fields_offset = len(self.bitsize_fields) + 1
    if not use_bitvector:
      self.bit_vector_encoding_dim = len(self.bitsize_fields) + 1
    else:
      self.bit_vector_encoding_dim = len(self.bitsize_fields) + 1 + self.bitsize
    self.feature_dim = NUM_FEATURES
    self.node_label_encoding_dim = len(self.arch_fields) + self.bit_vector_encoding_dim + self.feature_dim

    # Edge embedding translation
    if use_raw_edges:
      self.claripy_ops = sorted(list(expression_operations | backend_operations_all | backend_fp_operations | backend_strings_operations))
      self.custom_ops = ["reg_loc", "reg_data", "mem_loc", "mem_data"]
      self.all_ops = self.claripy_ops + self.custom_ops
    else:
      self.all_ops = REDUCED_OPS
    self.num_all_ops = len(self.all_ops) + 1
    self.num_edge_ops = self.num_all_ops * 2

    # Architectures
    self.archs = [ArchAMD64(), ArchX86(), ArchARM(), ArchMIPS32(), ArchAArch64()]

def encode_arch(arch_str: Optional[str], config: Config) -> Tensor:
  if config.use_arch:
    one_hot = [1.0 if arch_str == a else 0.0 for a in config.arch_fields]
    return torch.tensor(one_hot)
  else:
    return torch.tensor([0.0 for _ in config.arch_fields])

def encode_node_features(node_label: NodeLabel, config: Config) -> Tensor:
  # is float
  is_float = isinstance(node_label.ast_val, float)
  is_bool = isinstance(node_label.ast_val, bool)
  starts_with_f, many_f_s, msb = False, False, False
  is_zero, is_one, is_neg_one, is_two = False, False, False, False
  close_to_sp, is_very_small, is_small, is_large = False, False, False, False
  is_reg, is_artificial_reg, is_arg_reg, is_ret_reg = False, False, False, False
  has_addr = not (node_label.addr is None)

  # Different types of ast_val
  if is_float:
    pass
  elif is_bool:
    is_zero = not node_label.ast_val
    is_one = node_label.ast_val
  else:
    # Most significant bit
    hex_val = hex(node_label.ast_val)
    starts_with_f = "0xf" in hex_val
    many_f_s = len([c for c in hex_val if c == "f"]) >= 6
    msb = (node_label.ast_val >> (node_label.bitsize - 1)) & 1

    # Is zero
    is_zero = node_label.ast_val == 0
    is_one = node_label.ast_val == 1
    is_neg_one = node_label.ast_val == (1 << node_label.bitsize) - 1
    is_two = node_label.ast_val == 2

    # Close to arch initial stack pointer
    for arch in config.archs:
      close_to_sp = abs(node_label.ast_val - arch.initial_sp) < SP_RANGE

    # Size of the number
    is_very_small = len(hex_val) < 5
    is_small = len(hex_val) < 8
    is_large = node_label.ast_val > pow(2, node_label.bitsize / 4)

    # Argument Register info
    is_amd64_arg_reg = node_label.ast_val in ArchAMD64.argument_registers
    is_x86_arg_reg = node_label.ast_val in ArchX86.argument_registers
    is_mips_arg_reg = node_label.ast_val in ArchMIPS32.argument_registers
    is_arm32_arg_reg = node_label.ast_val in ArchARM.argument_registers
    is_arg_reg = is_amd64_arg_reg or is_x86_arg_reg or is_mips_arg_reg or is_arm32_arg_reg

    # Register
    for arch in config.archs:
      translated = arch.translate_register_name(node_label.ast_val)
      is_reg = is_reg or (translated in arch.registers)
      is_artificial_reg = is_artificial_reg or (node_label.ast_val in arch.artificial_registers_offsets)
      is_ret_reg = is_ret_reg or (node_label.ast_val == arch.ret_offset)

  # Assemble
  features = [
    is_float,
    is_bool,
    starts_with_f,
    many_f_s,
    msb,
    is_zero,
    is_one,
    is_neg_one,
    is_two,
    close_to_sp,
    is_very_small,
    is_small,
    is_large,
    is_arg_reg,
    is_reg,
    is_artificial_reg,
    is_ret_reg,
    has_addr,
  ]
  vector = [1.0 if x else 0.0 for x in features]
  return torch.tensor(vector)

def encode_bit_vector(node_label: NodeLabel, config: Config) -> Tensor:
  vector = config.bit_vector_encoding_dim * [0.0]
  node_label_bitsize = node_label.bitsize

  # Alignment features
  aligned = False
  for (i, bitsize) in enumerate(config.bitsize_fields):
    if node_label_bitsize == bitsize:
      vector[i] = 1.0
      aligned = True
  if not aligned:
    vector[config.unaligned_bitsize_index] = 1.0

  # Directly encode bitvector
  if config.use_bitvector:

    # Starting from the right hand side of the array, we do bitsizes
    bitsize_to_proc = min(config.bitsize, node_label_bitsize)
    offset = config.bitsize_fields_offset + config.bitsize - bitsize_to_proc
    value = node_label.ast_val
    for i in range(bitsize_to_proc):
      index = offset + bitsize_to_proc - i - 1
      # vector[index] = 1.0 if value & 1 else 0.0
      # value = value >> 1
      if isinstance(value, bool) or isinstance(value, int):
        vector[index] = 1.0 if value & 1 else 0.0
        value = value >> 1
      else:
        vector[index] = 0.0

  # Return
  return torch.tensor(vector)

def encode_edge_label(edge_label: Set[str], inverse: bool, config: Config) -> int:
  for edge_op in sorted(list(edge_label)):
    if edge_op in config.all_ops:
      idx = config.all_ops.index(edge_op)
    elif edge_op in REDUCE_OP_MAP:
      idx = config.all_ops.index(REDUCE_OP_MAP[edge_op])
    else:
      idx = len(config.all_ops)
    return idx if not inverse else idx + config.num_all_ops

def preproc_input(glow_input: GlowInput, config: Config) -> PreprocGlowInput:
  ast_graph = glow_input.ast_graph

  # Preprocess node labels
  arch = glow_input.arch if hasattr(glow_input, "arch") else None
  arch_tensor = encode_arch(arch, config)
  node_labels = []
  for node_id in ast_graph.graph.nodes:
    node_label = encode_bit_vector(ast_graph.node_to_label[node_id], config)
    node_features = encode_node_features(ast_graph.node_to_label[node_id], config)
    node_label_enc = torch.cat([arch_tensor, node_label, node_features], dim=0)
    node_labels.append(node_label_enc)
  node_labels = torch.stack(node_labels)

  # Preprocess edges & edge labels
  edge_labels = []
  edge_tensors = []
  for edge in ast_graph.graph.edges:
    (s, d) = edge

    # Forward edge & edge label
    edge_tensors.append(torch.tensor([s, d], dtype=torch.long))
    edge_labels.append(encode_edge_label(ast_graph.edge_to_labels[edge], False, config))

    # Backward edge & edge label
    edge_tensors.append(torch.tensor([d, s], dtype=torch.long))
    edge_labels.append(encode_edge_label(ast_graph.edge_to_labels[edge], True, config))
  edge_labels = torch.tensor(edge_labels)
  edges = torch.transpose(torch.stack(edge_tensors), 0, 1)

  # Preprocess var nodes
  var_nodes = [list(v.nodes) for v in glow_input.vars]
  # var_nodes = [[1, 2, 3], [3, 4, 5], [2, 3]]
  # var_nodes = [[1], [2], [3], [3], [4], ...]

  # Flatten
  if config.flatten_vars:
    var_nodes = [[n] for nodes in var_nodes for n in nodes]

  # Return the preprocessed input
  return PreprocGlowInput(node_labels, edge_labels, edges, var_nodes)

def preproc_output(input: GlowInput, output: GlowOutput, config: Config, fmt: str) -> Tensor:
  if fmt == 'onehot':
    type_tensors = [config.type_set.type_to_tensor(ty) for ty in output.types]
  else:
    type_tensors = [config.type_set.index_of_type(ty) for ty in output.types]

  # Flatten
  if config.flatten_vars:
    type_tensors = [t for (t, v) in zip(type_tensors, input.vars) for _ in range(len(v.nodes))]
  if fmt == 'onehot':
    return torch.stack(type_tensors)
  return torch.LongTensor(type_tensors)
