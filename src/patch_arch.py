import pickle

def setup_parser(parser):
  parser.add_argument("input", type=str)
  parser.add_argument("arch", type=str)

def main(args):
  dataset = pickle.load(open(args.input, "rb"))
  for (i, o) in dataset:
    i.arch = args.arch
  pickle.dump(dataset, open(args.input, "wb"))
