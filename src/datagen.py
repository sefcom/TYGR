from argparse import ArgumentParser
import os
import pickle

from .methods import get_method, DEFAULT_METHOD

def setup_parser(parser: ArgumentParser):
  parser.add_argument("input", type=str)
  parser.add_argument("output", type=str)
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("--parallel", action="store_true")
  parser.add_argument("--no-splice", action="store_true")
  parser.add_argument("--ignore-functions", type=str, default="src/data/ignore_functions.json")
  parser.add_argument("--output-functions", type=str, default=None)
  parser.add_argument("--max-input-kbs", type=int, default=150*1024*1024) # 150 MB

def main(args):

  # Check if under max file size
  size_kbs = os.path.getsize(args.input)
  if size_kbs > args.max_input_kbs:
    print(f"{args.input} is more than {args.max_input_kbs} KBs!")
    exit(1)

  method = get_method(args.method, args, phase="datagen")
  ## let us filter here
  # method.generate(args.input, args.output)
  dataset = method.generate(args.input, args.output)
  dataset = method.filter_ill_formed(dataset)
  with open(args.output, "wb") as f:
    pickle.dump(dataset, f)
