from argparse import ArgumentParser

from .method import Method
from .flow import Flow, setup_parser as flow_setup_parser
from .glow import Glow, setup_parser as glow_setup_parser
from .glow_th import GlowTH, setup_parser as glow_th_setup_parser

DEFAULT_METHOD = "glow"

def get_method(name: str, args, phase) -> Method:
  if name == "flow":
    return Flow(args, phase)
  elif name == "glow":
    return Glow(args, phase)
  elif name == "glow-th":
    return GlowTH(args, phase)
  else:
    raise Exception(f"Unknown method {name}")

def setup_parser(parser: ArgumentParser):
  flow_setup_parser(parser)
  glow_setup_parser(parser)
  glow_th_setup_parser(parser)
