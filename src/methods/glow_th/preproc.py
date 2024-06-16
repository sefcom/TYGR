from typing import *
import torch
import random
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
from ..glow.common import GlowInput, GlowOutput, GlowVar
from ..glow.preproc import encode_arch, encode_bit_vector, encode_node_features, encode_edge_label, NUM_FEATURES
from .common import PreprocGlowTHInput

class Config:
  def __init__(
    self,
    type_set: str = "std",
    bitsize: int = 32,
    use_bitvector: bool = False,
    use_raw_edges: bool = False,
    flatten_vars: bool = True,
    use_arch: bool = True,
    type_hint_rate = None,
    type_hint_limit = None,
  ):
    self.use_bitvector = use_bitvector
    self.flatten_vars = flatten_vars
    self.use_arch = use_arch
    self.type_hint_rate = type_hint_rate
    self.type_hint_limit = type_hint_limit

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
    self.type_dim = self.type_set.num_types()
    self.feature_dim = NUM_FEATURES
    self.node_label_encoding_dim = len(self.arch_fields) + self.bit_vector_encoding_dim + self.feature_dim + self.type_dim

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

def encode_node_type(node_id: int, vars: List[GlowVar], types: List[Type], config: Config) -> Tensor:
  for (i, var) in enumerate(vars):
    if node_id in var.nodes:
      ty = types[i]
      return config.type_set.type_to_tensor(ty)
  return torch.zeros(config.type_set.num_types())

def preproc_input(i: int, glow_input: GlowInput, glow_output: GlowOutput, config: Config) -> PreprocGlowTHInput:
  ast_graph = glow_input.ast_graph

  # Keep ids
  if config.type_hint_rate is not None:
    to_keep_ids = [i for i in range(len(glow_input.vars)) if random.random() < config.type_hint_rate]
  elif config.type_hint_limit is not None:
    to_keep_ids = [i for i in range(len(glow_input.vars))]
    random.shuffle(to_keep_ids)
    to_keep_ids = to_keep_ids[:config.type_hint_limit]
  other_vars = [glow_input.vars[i] for i in to_keep_ids]
  other_types = [glow_output.types[i] for i in to_keep_ids]

  # Preprocess node labels
  arch = glow_input.arch if hasattr(glow_input, "arch") else None
  arch_tensor = encode_arch(arch, config)
  node_labels = []
  for node_id in ast_graph.graph.nodes:
    node_label = encode_bit_vector(ast_graph.node_to_label[node_id], config)
    node_features = encode_node_features(ast_graph.node_to_label[node_id], config)
    node_type = encode_node_type(node_id, other_vars, other_types, config)
    node_label_enc = torch.cat([arch_tensor, node_label, node_features, node_type], dim=0)
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
  var_nodes = [glow_input.vars[i].nodes]

  # Flatten
  if config.flatten_vars:
    var_nodes = [[n] for nodes in var_nodes for n in nodes]

  # Return the preprocessed input
  return PreprocGlowTHInput(node_labels, edge_labels, edges, var_nodes)

def preproc_output(i: int, input: GlowInput, output: GlowOutput, config: Config, fmt: str) -> Tensor:
  ty = output.types[i]
  if fmt == 'onehot':
    type_tensors = [config.type_set.type_to_tensor(ty)]
  else:
    type_tensors = [config.type_set.index_of_type(ty)]

  # Flatten
  if config.flatten_vars:
    type_tensors = type_tensors * len(input.vars[i].nodes)

  if fmt == 'onehot':
    return torch.stack(type_tensors)

  return torch.LongTensor(type_tensors)
