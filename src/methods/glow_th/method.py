from typing import *
import pickle
import copy

from torch import nn, Tensor

from ...analysis.types import TypeSet
from ...analysis.types.btypes import *
from ...analysis.dwarf.location import *
from ..glow.common import GlowInput, GlowOutput
from ..glow import model, accuracy
from .common import PreprocGlowTHInput
from . import preproc, dataset
from ..method import Method

class GlowTH(Method[GlowInput, PreprocGlowTHInput, GlowOutput, Tensor]):
  def __init__(self, args, phase=None):
    super().__init__(args, phase=None)

    # Prepare preprocessor configuration if we are in learning phase
    if phase == "train" or phase == "test":
      self.loss_fn = nn.CrossEntropyLoss()
      self.preproc_config = preproc.Config(
        type_set=self.args.type_set,
        use_bitvector=self.args.glow_use_bitvector,
        bitsize=self.args.glow_bitvector_size,
        type_hint_rate=self.args.glow_th_rate,
        type_hint_limit=self.args.glow_th_limit,
      )

    # Prepare loss function and model configurations during training
    if phase == "train":
      self.model_config = model.Config(
        self.preproc_config,
        node_latent_dim = 64,
        num_msg_pass_layers = 8,
      )

  def filter_ill_formed(self, list_samples):
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
    preproc_num_functions = len(result)
    print("Source #functions:", source_num_functions)
    print("Well Formed #functions:", preproc_num_functions)
    # print("Source #vars:", source_num_vars)
    # print("Well Formed #vars:", preproc_num_vars)

    return result

  def preproc_input(self, i: int, input: GlowInput, output: GlowOutput) -> List[PreprocGlowTHInput]:
    return preproc.preproc_input(i, input, output, self.preproc_config)

  def preproc_output(self, i: int, input: GlowInput, output: GlowOutput, fmt: str='onehot') -> List[Tensor]:
    return preproc.preproc_output(i, input, output, self.preproc_config, fmt)

  def dataset(self, ds) -> dataset.GlowTHDataset:
    return dataset.GlowTHDataset(ds, self, keep_rate=self.args.glow_th_keep_sample)

  def collate_function(self):
    return dataset.collate_fn

  def model(self) -> nn.Module:
    return model.GlowGNN(self.model_config)

  def loss(self, y_pred: Tensor, o) -> Tensor:
    (y_ids, _) = o
    return self.loss_fn(y_pred, y_ids)

  def accuracy(self, y_pred: Tensor, o: Tensor) -> Tuple[int, int]:
    (_, y) = o
    return accuracy.accuracy(self.preproc_config, y_pred, y)

  def topk_accuracy(self, k: int, y_pred: Tensor, o: Tensor) -> Tuple[int, int]:
    (_, y) = o
    return accuracy.topk_accuracy(self.preproc_config, k, y_pred, y)

  def iter_output(self, y_pred: Tensor, y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
    (n, _) = y_pred.size()
    for i in range(n):
      yield (y_pred[i], y[i])

  def type_set(self) -> TypeSet:
    return self.preproc_config.type_set
