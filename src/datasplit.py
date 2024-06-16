import pickle
import random

def setup_parser(parser):
  parser.add_argument("input", type=str)
  parser.add_argument("--train", type=str, default="data/train.pkl")
  parser.add_argument("--validation", type=str, default="data/validation.pkl")
  parser.add_argument("--test", type=str, default="data/test.pkl")
  parser.add_argument("--train-perc", type=float, default="0.8", help="The percentage of datapoints going into training set")
  parser.add_argument("--validation-perc", type=float, default="0.1", help="The percentage of datapoints going into validation set")
  parser.add_argument("--test-perc", type=float, default="0.1", help="The percentage of datapoints going into testing set")
  parser.add_argument("-s", "--seed", type=int, default=135792468)
  parser.add_argument("-v", "--verbose", action="store_true")

def main(args):
  if args.verbose:
    print("Spliting into train, validation, and test sets...")

  # First load the dataset
  dataset = pickle.load(open(args.input, "rb"))

  # Setup random seed and shuffle the dataset
  random.seed(args.seed)
  random.shuffle(dataset)

  # Get the size of each dataset
  size = len(dataset)
  num_train = int(args.train_perc * size)
  num_validation = int(args.validation_perc * size)
  #num_test = int(args.test_perc * size)
  num_test = size - num_train - num_validation
  if args.verbose:
    print(f"#Train: {num_train}, #Validation: {num_validation}, #Test: {num_test}")
    
  # Dump the splitted datasets
  pickle.dump(dataset[0:num_train], open(args.train, "wb"))
  pickle.dump(dataset[num_train:num_train + num_validation], open(args.validation, "wb"))
  pickle.dump(dataset[num_train + num_validation:], open(args.test, "wb"))
