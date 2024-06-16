from typing import *
from argparse import ArgumentParser
import pickle

import torch
from torch import Tensor

from .methods import get_method, setup_parser as methods_setup_parser, Method, DEFAULT_METHOD
from .analysis.types import Type
from .analysis.types.type_set import DEFAULT_TYPE_SET_CLASS
from .analysis.utils.histogram import Histogram
from .utils.cuda import setup_cuda
from .utils import learn

def setup_parser(parser: ArgumentParser):
  parser.add_argument("model", type=str)
  parser.add_argument("test_set", type=str)
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
  parser.add_argument("-g", "--gpu", type=int, default=None)
  parser.add_argument("-k", type=int, default=3)
  parser.add_argument("--type-set", type=str, default=DEFAULT_TYPE_SET_CLASS)
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("-a", "--analysis", action="store_true")
  parser.add_argument("--batch_size", type=int, default=32)

  methods_setup_parser(parser)

class AnalysisContext:
  def __init__(self):
    self.map: Dict[Type, Histogram] = {}

  def add_datapoint(self, expected: Type, predicted: Type):
    if expected not in self.map:
      self.map[expected] = Histogram(str(expected))
    self.map[expected].add(predicted)

  def print(self):
    print(self.map)

def analysis_callback(method: Method, ctx: AnalysisContext, k: int, y_pred: Tensor, y: Tensor):
  for (pred_type_tensor, gt_type_tensor) in method.iter_output(y_pred, y):
    pred_type = method.tensor_to_type(pred_type_tensor)
    pred_topk_types = method.tensor_to_topk_types(k, pred_type_tensor)
    gt_type = method.tensor_to_type(gt_type_tensor)

    # Print
    print(f"|- Expected: {gt_type}, Predicted: {pred_type}, Top-{k} Predicted: {pred_topk_types}")

    # Add datapoint
    ctx.add_datapoint(gt_type, pred_type)

def main(args):
  # GPU
  setup_cuda(args.gpu)
  # Load model
  model = torch.load(args.model)

  # Method
  method = get_method(args.method, args, phase="test")

  # Testing dataset
  dataset = pickle.load(open(args.test_set, "rb"))
  dataset = method.filter_ill_formed(dataset)

  # Analysis
  if args.analysis:
    anal_ctx = AnalysisContext()
    callback = lambda y_p, y: analysis_callback(method, anal_ctx, args.k, y_p, y)
    progress_bar = False
    prompt = None
  else:
    callback = None
    progress_bar = True
    prompt = "[Test]"

  # Evaluate
  model.eval()
  result = learn.run(method, model, dataset, args, prompt=prompt, callback=callback, progress_bar=progress_bar)

  # Analysis
  if args.analysis:
    anal_ctx.print()
    result.print()
