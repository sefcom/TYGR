def setup_parser(parser):
  parser.add_argument("--glow-th-keep-sample", type=float, default=0.3)
  parser.add_argument("--glow-th-rate", type=float)
  parser.add_argument("--glow-th-limit", type=int)
