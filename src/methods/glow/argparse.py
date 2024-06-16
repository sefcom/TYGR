from argparse import ArgumentParser

def setup_parser(parser: ArgumentParser):
  parser.add_argument("--glow-use-bitvector", action="store_true")
  parser.add_argument("--glow-no-arch", action="store_true")
  parser.add_argument("--glow-bitvector-size", type=int, default=32)
  parser.add_argument("--glow-num-encoder-layers", type=int, default=1)
  parser.add_argument("--glow-latent-dim", type=int, default=64)
  parser.add_argument("--glow-num-msg-pass-layers", type=int, default=8)
  parser.add_argument("--glow-no-share-weight", action="store_true")
  parser.add_argument("--glow_dropout", type=float, default=0)
  parser.add_argument("--glow_decoder_type", type=str, default='independent')
  parser.add_argument("--glow_beam_size", type=int, default=4)
