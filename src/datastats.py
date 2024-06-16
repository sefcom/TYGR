from argparse import ArgumentParser
import pickle

from .methods import get_method, DEFAULT_METHOD

def setup_parser(parser: ArgumentParser):
  parser.add_argument("dataset", type=str)
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)

def main(args):
  method = get_method(args.method, args, phase="inspect")
  dataset = pickle.load(open(args.dataset, "rb"))
  method.stats(dataset)
