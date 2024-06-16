from typing import *
from torch import nn, Tensor

from .common import FlowInput, PreprocFlowInput
from . import datagen
from ..method import Method
from ...analysis.types import Type
from .argparse import setup_parser

class Flow(Method[FlowInput, PreprocFlowInput, Type, Tensor]):
  def __init__(self, args, phase):
    super().__init__(args, phase)

  def generate(self, input_file_name: str) -> List[Tuple[FlowInput, Type]]:
    options = datagen.Options(verbose=self.args.verbose)
    dataset = list(datagen.generate_flow_dataset(input_file_name, options))
    return dataset

  def preproc_input(self, input: FlowInput) -> PreprocFlowInput:
    raise Exception("Not implemented")

  def preproc_output(self, output: Type) -> Tensor:
    raise Exception("Not implemented")

  def model(self) -> nn.Module:
    raise Exception("Not implemented")

  def loss(self, y_pred: Tensor, y: Tensor) -> Tensor:
    raise Exception("Not implemented")

  def accuracy(self, y_pred: Tensor, y: Tensor) -> Tuple[int, int]:
    raise Exception("Not implemented")

  def topk_accuracy(self, k: int, y_pred: Tensor, y: Tensor) -> Tuple[int, int]:
    raise Exception("Not implemented")
