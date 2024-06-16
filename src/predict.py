from argparse import ArgumentParser
import pickle

import torch
from torch import Tensor

from .methods import get_method, setup_parser, DEFAULT_METHOD
from .analysis.types import Type
from .analysis.types.type_set import DEFAULT_TYPE_SET_CLASS
from .utils.cuda import setup_cuda


def setup_parser(parser: ArgumentParser):
  parser.add_argument("model", type=str)
  parser.add_argument("input_file", type=str)
  parser.add_argument("output_file", type=str)
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("--parallel", action="store_true")
  parser.add_argument("--no-splice", action="store_true")
  parser.add_argument("--ignore-functions", type=str, default="src/data/ignore_functions.json")
  parser.add_argument("--output-functions", type=str, default=None)
  parser.add_argument("-g", "--gpu", type=int, default=None)
  parser.add_argument("--type-set", type=str, default=DEFAULT_TYPE_SET_CLASS)


def main(args):

  # GPU
  setup_cuda(args.gpu)

  # Load model
  model = torch.load(args.model)

  # Generate the input data
  method = get_method(args.method, args, phase="predict")

  method.predict_and_bickle(model, args.input_file, args.output_file)


