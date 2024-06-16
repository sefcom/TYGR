from typing import *
from torch import nn, Tensor

from ..analysis.types import TypeSet

I = TypeVar("I")
PI = TypeVar("PI")
O = TypeVar("O")
PO = TypeVar("PO")

class Method(Generic[I, PI, O, PO]):
  """
  A generic method for binary analysis
  """

  def __init__(self, args, phase):
    """
    Initialize a method. Args are provided as command line arguments.
    Phase could be "datagen", "train", and "test". A method will need
    to initialize the environment differently according to these phases
    """
    self.args = args
    self.phase = phase

  def generate(self, input_file_name: str) -> List[Tuple[I, O]]:
    """
    The generate function will take in an input binary file and generate
    a list of (input, output) pairs.
    """
    raise Exception("Not implemented")

  def stats(self, dataset: List[Tuple[I, O]]):
    """
    Print the statistics of the dataset
    """
    raise Exception("Not implemented")

  def visualize(self, base_file_name: str, datapoint: Tuple[I, O]):
    """
    Visualize the given datapoint and store it into the given base file
    name.
    Note that it is a base file name rather than full file name. A
    visualizer might choose to output multiple files corresponding to
    the datapoint. They should all use the same base file name with
    additional suffixes.
    """
    raise Exception("Not implemented")

  def generate_inputs(self, input_file_name: str):
    """
    Generate only the inputs; relevant for prediction.
    """
    raise Exception("Not implemented")

  def preproc_input(self, input: I, output: O) -> PI:
    """
    Preprocess an input and produce a processed input that is ready to
    be fed into the model
    """
    raise Exception("Not implemented")

  def preproc_output(self, input: I, output: O) -> PO:
    """
    Preprocess an output and produce a processed output that is ready to
    be passed into the loss function
    """
    raise Exception("Not implemented")

  def model(self) -> nn.Module:
    """
    Return a brand new model of this method
    """
    raise Exception("Not implemented")

  def loss(self, y_pred: PO, y: PO) -> Tensor:
    """
    Calculate the loss based on prediction and ground truth
    - y_pred: predicted result
    - y: ground truth result
    """
    raise Exception("Not implemented")

  def accuracy(self, y_pred: PO, y: PO) -> Tuple[int, int]:
    """
    Calculate the accuracy based on prediction and ground truth
    - y_pred: predicted result
    - y: ground truth result
    Returns
    - A tuple containing
      - [0]: the total number of predictions
      - [1]: amount of correct predictions
    """
    raise Exception("Not implemented")

  def topk_accuracy(self, k: int, y_pred: PO, y: PO) -> Tuple[int, int]:
    """
    Calculate the top-k accuracy based on prediction and ground truth
    - k: the k of top-k
    - y_pred: predicted result
    - y: ground truth result
    Returns
    - A tuple containing
      - [0]: the total number of predictions
      - [1]: amount of correct predictions
    """
    raise Exception("Not implemented")

  def type_set(self) -> TypeSet:
    raise Exception("Not implemented")

  def iter_output(self, y_pred: PO, y: PO) -> Iterator[Tuple[Tensor, Tensor]]:
    raise Exception("Not implemented")

  def tensor_to_type(self, t: Tensor) -> Type:
    return self.type_set().tensor_to_type(t)

  def tensor_to_topk_types(self, k: int, t: Tensor) -> List[Type]:
    return self.type_set().tensor_to_topk_types(t, k)
