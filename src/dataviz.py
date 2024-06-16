from argparse import ArgumentParser
import random
import pickle

from .methods import get_method, DEFAULT_METHOD

def setup_parser(parser: ArgumentParser):
  parser.add_argument("dataset", type=str)
  parser.add_argument("outdir", type=str)
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("-b", "--begin")
  parser.add_argument("-e", "--end")
  parser.add_argument("-a", "--amount")
  parser.add_argument("-i", "--i")
  parser.add_argument("-r", "--random")
  parser.add_argument("-s", "--seed", default=1235678)

def main(args):
  method = get_method(args.method, args, phase="dataviz")
  dataset = pickle.load(open(args.dataset, "rb"))
  source_indices = list(range(len(dataset)))

  # Set seed
  if args.random:
    random.seed(args.seed)
    random.shuffle(source_indices)

  # Compute indices
  indices = []
  if args.i:
    indices.append(source_indices[args.i])
  elif args.begin:
    if args.amount:
      for i in range(args.amount):
        indices.append(source_indices[args.begin + i])
    elif args.end:
      for i in range(args.begin, args.end):
        indices.append(source_indices[i])
    else:
      for i in range(args.i, len(dataset)):
        indices.append(source_indices[i])
  else:
    for i in range(len(dataset)):
      indices.append(source_indices[i])

  # Output
  for i in indices:
    base_file_name = f"{args.outdir}/{i}"
    method.visualize(base_file_name, dataset[i])
