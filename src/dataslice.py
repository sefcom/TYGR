import pickle
import random

def setup_parser(parser):
  parser.add_argument("input", type=str)
  parser.add_argument("output", type=str)
  parser.add_argument("-r", "--random", action="store_true")
  parser.add_argument("-s", "--seed", type=int, default=12345678)
  parser.add_argument("-b", "--begin", type=int, default=0)
  parser.add_argument("-e", "--end", type=int)
  parser.add_argument("-a", "--amount", type=int, default=0)
  parser.add_argument("-v", "--verbose", action="store_true")

def main(args):
  # Load the dataset
  dataset = pickle.load(open(args.input, "rb"))
  indices = list(range(len(dataset)))

  # Randomize
  if args.random:
    random.seed(args.seed)
    random.shuffle(indices)

  # Get the relavent indices
  if args.end:
    indices = indices[args.begin:args.end]
  elif args.amount:
    indices = indices[args.begin:args.begin + args.amount]
  else:
    print("Must specify --end (-e) or --amount (-a)")
    exit(1)

  # Generate sliced dataset based on indices
  indices = sorted(indices)
  sliced_dataset = [dataset[i] for i in indices]

  # Dump the sliced dataset
  pickle.dump(sliced_dataset, open(args.output, "wb"))
