from argparse import ArgumentParser
import pickle
import random

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .methods import get_method, setup_parser as methods_setup_parser, DEFAULT_METHOD
from .analysis.types.type_set import DEFAULT_TYPE_SET_CLASS
from .utils.cuda import setup_cuda
from .utils import learn

def setup_parser(parser: ArgumentParser):
  parser.add_argument("train_set", type=str)
  parser.add_argument("valid_set", type=str)
  parser.add_argument("-o", "--output", type=str, default="model/model")
  parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD)
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("-e", "--epoch", type=int, default=35)
  parser.add_argument("-k", type=int, default=3)
  parser.add_argument("-s", "--seed", type=int, default=12345678)
  parser.add_argument("--no-shuffle", action="store_true")
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--type-set", type=str, default=DEFAULT_TYPE_SET_CLASS)
  parser.add_argument("-g", "--gpu", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--optimizer", type=str, default="adam")

  methods_setup_parser(parser)

def get_optimizer(args, model: nn.Module) -> optim.Optimizer:
  if args.optimizer == "sgd":
    return optim.SGD(model.parameters(), lr=args.lr)
  elif args.optimizer == "adam":
    return optim.Adam(model.parameters(), lr=args.lr)

def main(args):
  # GPU
  setup_cuda(args.gpu)

  # Randomness
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Method
  method = get_method(args.method, args, phase="train")

  # Training & Validation dataset
  train_set = pickle.load(open(args.train_set, "rb"))
  train_set = method.filter_ill_formed(train_set)
  valid_set = pickle.load(open(args.valid_set, "rb"))
  valid_set = method.filter_ill_formed(valid_set)

  # Model
  model = method.model()

  # Optimizer
  optimizer = get_optimizer(args, model)
  scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, min_lr=1e-6, factor=0.1, verbose=True)

  # Best Performance
  minimal_loss = 1000000000

  # Epoch list
  for e in range(args.epoch):
    # Train
    model.train()
    # print("Enter Train")
    #print(train_set)
    learn.run(method, model, train_set, args, prompt=f"[Train] Epoch {e}", optimizer=optimizer, shuffle=not args.no_shuffle)

    # Validate
    model.eval()
    with torch.no_grad():
      # print("Enter Validation")
      validation_result = learn.run(method, model, valid_set, args, prompt=f"[Valid] Epoch {e}")
    scheduler.step(validation_result.accuracy)

    # Save the best performing model
    if validation_result.loss < minimal_loss:
      minimal_loss = validation_result.loss
      print(f"save best model epoch{e}")
      torch.save(model, f"{args.output}.best.model")

    print(f"save epoch{e} model")
    torch.save(model, f"{args.output}.last.model")


  # # Save the last epoch model
  # torch.save(model, f"{args.output}.last.model")
