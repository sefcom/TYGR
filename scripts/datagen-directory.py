import argparse
import sys
import os
import json
import pickle

command = "bityr"

initial_ign_fns = [
  "xpalloc",
  "z85_encode",
  "try_tempname_len",
]

def parser():
  parser = argparse.ArgumentParser(description='Data Generation')
  setup_parser(parser)
  return parser

def setup_parser(parser):
  parser.add_argument("input_dir", type=str)
  parser.add_argument("output_dir", type=str)
  parser.add_argument("-o", "--overwrite", action="store_true")
  parser.add_argument("--merge-file", type=str, default="merged.pkl")
  parser.add_argument("--split-dir", type=str, default="splits")
  parser.add_argument("--train", type=str, default="train.pkl")
  parser.add_argument("--validation", type=str, default="validation.pkl")
  parser.add_argument("--test", type=str, default="test.pkl")
  parser.add_argument("--method", type=str, default="glow")
  parser.add_argument("--exit-when-failed", action="store_true")
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("-p", "--parallel", action="store_true")
  parser.add_argument("--ignore-functions-file", type=str, default=".ign_fns.json")
  parser.add_argument("--output-functions-file", type=str, default=".tmp_ign_fns.json")
  parser.add_argument("--symbolize-init-pointers", action="store_true")

def main(args):
  processed = []
  ign_fns = [ign_fn for ign_fn in initial_ign_fns]
  json.dump(ign_fns, open(args.ignore_functions_file, "w"))

  binaries = list(os.listdir(args.input_dir))

  # Process all the files
  num_bins = len(binaries)
  for count, bin in enumerate(binaries):
    output_name = f"{bin}.pkl"
    item_path = os.path.join(args.input_dir, bin)
    output_path = os.path.join(args.output_dir, output_name)

    # Check if is bincutable
    # is_bin = os.access(item_path, os.X_OK)
    is_bin = True
    generated = os.path.exists(output_path)
    if is_bin:
      print(f"{count + 1}/{num_bins} Processing {bin}... ", end="")
      sys.stdout.flush()

      if generated and not args.overwrite:
        print("Generated. Skipping")
        generated_file_name = output_path
      else:
        cmd = f"./{command} datagen {item_path} {output_path} -m {args.method} \
                --ignore-functions {args.ignore_functions_file} \
                --output-functions {args.output_functions_file}"

        if args.parallel:
          cmd += " --parallel"

        if args.verbose:
          cmd += " -v"

        if args.symbolize_init_pointers:
          cmd += " --symbolize-init-pointers"

        err = os.system(cmd)
        if not err:
          print("Success")
          generated_file_name = output_path
        else:
          print("Failed")
          generated_file_name = None
          if args.exit_when_failed:
            return

      if generated_file_name is not None:
        processed.append(generated_file_name)

        new_ign_fns = json.load(open(args.output_functions_file, "r"))
        ign_fns += new_ign_fns
        json.dump(ign_fns, open(args.ignore_functions_file, "w"))

  # Merge all of them
  merge_path = os.path.join(args.output_dir, args.merge_file)
  datasets = " ".join(processed)
  os.system(f"./{command} datamerge {datasets} -o {merge_path}")

  # Create split
  split_path = os.path.join(args.output_dir, "splits")
  train_path = os.path.join(split_path, args.train)
  validation_path = os.path.join(split_path, args.validation)
  test_path = os.path.join(split_path, args.test)
  os.system(f"mkdir -p {split_path}")
  os.system(f"./{command} datasplit {merge_path} \
              --train {train_path} \
              --validation {validation_path} \
              --test {test_path}")

if __name__ == "__main__":
  args = parser().parse_args()
  main(args)
