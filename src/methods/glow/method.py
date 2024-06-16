from typing import *
import pickle
import copy
import random

from torch import nn, Tensor

from ...analysis.types import TypeSet
from ...analysis.types.btypes import *
from ...analysis.dwarf.location import *
from .common import GlowInput, PreprocGlowInput, GlowOutput
from . import datagen, visualize, preproc, model, accuracy, stats, predict, dataset
from ..method import Method

class Glow(Method[GlowInput, PreprocGlowInput, GlowOutput, Tensor]):
  def __init__(self, args, phase=None):
    super().__init__(args, phase=None)

    # Prepare data generation options if the phase is datagen
    if phase == "datagen":
      self.datagen_options = datagen.Options(
        verbose=self.args.verbose,
        parallel=self.args.parallel,
        no_splice=self.args.no_splice,
        ignore_functions_file=self.args.ignore_functions,
        output_functions=self.args.output_functions,
      )

    # Prepare preprocessor configuration if we are in learning phase
    if phase == "train" or phase == "test":
      self.loss_fn = nn.CrossEntropyLoss()
      self.preproc_config = preproc.Config(
        type_set=self.args.type_set,
        use_bitvector=self.args.glow_use_bitvector,
        bitsize=self.args.glow_bitvector_size,
        use_arch=not self.args.glow_no_arch,
      )

    # Prepare loss function and model configurations during training
    if phase == "train":
      self.model_config = model.Config(self.preproc_config,
                                       decoder_type=args.glow_decoder_type,
                                       dropout_rate=args.glow_dropout,
                                       beam_size=args.glow_beam_size)

    if phase == "predict":
      self.datagen_options = datagen.Options(
        verbose=self.args.verbose,
        parallel=self.args.parallel,
        no_splice=self.args.no_splice,
        ignore_functions_file=self.args.ignore_functions,
        output_functions=self.args.output_functions,
        predict_phase=True,
      )

      self.preproc_config = preproc.Config(
        type_set=self.args.type_set,
      )

  def generate(self, input_file_name: str, out_file_dir: str) -> List[Tuple[GlowInput, GlowOutput]]:
    dataset = list(datagen.generate_glow_dataset(input_file_name, out_file_dir, self.datagen_options))
    return dataset

  def stats(self, dataset: List[Tuple[GlowInput, GlowOutput]]):
    print(stats.stats(dataset))

  def visualize(self, base_file_name: str, datapoint: Tuple[GlowInput, GlowOutput]):
    visualize.visualize(base_file_name, datapoint)

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
    print("Source #vars:", source_num_vars)
    print("Well Formed #vars:", preproc_num_vars)

    return result

  def generate_inputs(self, input_file_name: str):
    return predict.generate_glow_inputs(input_file_name, self.datagen_options)

  def preproc_input(self, input: GlowInput, _: GlowOutput) -> PreprocGlowInput:
    return preproc.preproc_input(input, self.preproc_config)

  def preproc_output(self, input: GlowInput, output: GlowOutput, fmt: str='onehot') -> Tensor:
    return preproc.preproc_output(input, output, self.preproc_config, fmt)

  def dataset(self, ds) -> dataset.GlowDataset:
    return dataset.GlowDataset(ds, self)

  def preproc_fine_tuning(self, dataset: List[Tuple[GlowInput, GlowOutput]]) -> Tuple[List[Tuple[GlowInput, GlowOutput]], List[Tuple[GlowInput, GlowOutput]]]:
    train_set, test_set = [], []

    for (i, o) in dataset:
      if len(i.vars) > self.args.type_hint_count:
        var_types = list(zip(i.vars, o.types))
        random.shuffle(var_types)

        # Generate train var types and datapoint
        train_var_types = var_types[:self.args.type_hint_count]
        train_vars, train_types = [vs for (vs, _) in train_var_types], [ts for (_, ts) in train_var_types]
        train_input = GlowInput(i.input_file_name, i.directory, i.file_name, i.function_name, i.low_high_pc, i.ast_graph, train_vars, i.arch)
        train_output = GlowOutput(train_types)
        train_set.append((train_input, train_output))

        # Generate test var types and datapoint
        test_var_types = var_types[self.args.type_hint_count - 1:]
        test_vars, test_types = [vs for (vs, _) in test_var_types], [ts for (_, ts) in test_var_types]
        test_input = GlowInput(i.input_file_name, i.directory, i.file_name, i.function_name, i.low_high_pc, i.ast_graph, test_vars, i.arch)
        test_output = GlowOutput(test_types)
        test_set.append((test_input, test_output))

    return (train_set, test_set)

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

  def predict_and_bickle(self, model, input_file_name: str, bickle_file: str):
    inputs = self.generate_inputs(input_file_name)

    var_dict = dict()

    for inp in inputs:
      subprog_low_pc = inp.low_high_pc[0]
      x = self.preproc_input(inp, None)
      num_vars, node_labels, edge_labels, edges, var_gather, var_scatter = predict.collate_inputs([x])
      y_pred, loss = model(num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, None)
      (n, _) = y_pred.size()

      btype_dict = dict()
      if subprog_low_pc in var_dict:
        btype_dict = var_dict[subprog_low_pc]

      if self.datagen_options.verbose:
        print(f"lowpc = {hex(subprog_low_pc)}")

      for i in range(n):
        var = inp.vars[i]
        locs = var.locs
        ty = self.preproc_config.type_set.tensor_to_type(y_pred[i])
        btype = type_to_btype(ty)

        if self.datagen_options.verbose:
          print(f"\tty={ty} \t bty={btype}")

        # For the predictions, we care only about their lowpc values,
        # which correspond to the place at which a prediction occurred
        for (low_pc, _, dwloc) in locs:
          if isinstance(dwloc, CfaLocation):
            predict.add_to_set_valued(btype_dict, ("cfa", dwloc.arg), ("pc", low_pc, btype))

          elif isinstance(dwloc, RegLocation):
            predict.add_to_set_valued(btype_dict, ("reg", dwloc.reg_num), ("pc", low_pc, btype))

          elif isinstance(dwloc, AddrLocation):
            predict.add_to_set_valued(btype_dict, ("addr", dwloc.arg), ("pc", low_pc, btype))

        var_dict[subprog_low_pc] = copy.deepcopy(btype_dict)

    # Write to the bickle file
    with open(bickle_file, "wb") as f:
      pickle.dump(var_dict, f)
      f.close()
