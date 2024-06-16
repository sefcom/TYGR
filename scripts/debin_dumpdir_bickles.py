'''
Run this to get debin to predict things

This is done in two parts and should be run in interactive mode.
The commands are specialized to the file structure of this Docker.

    python3 -i ~/bityr/scripts/debin_dumpdir_bickle.py \
        -idir /debin/my-exp/stripped \
        -pdir /debin/my-exp/predicted \
        -odir /debin/my-exp/bickle

The script first tries to start Nice2Predict. Wait for it to initialize, then run

    run(args)

'''

import argparse
import os
import sys
import datetime

# Reset processes with "debin" in their name ... and will probably also close vim
def reset():
  pass
  # os.system("pgrep -f debin | xargs kill -9")

''' SAMPLE COMMAND
./bazel-bin/n2p/json_server/json_server \
    --port 8604 \
    --model ../models/crf/x86/model \
    --valid_labels ../c_valid_labels \
    -logtostderr &
'''
# Start the Nice2Predict server
def start_n2p(args):
  server = os.path.join(args.debin_dir, "Nice2Predict/bazel-bin/n2p/json_server/json_server")
  valid_labels = os.path.join(args.debin_dir, "c_valid_labels")
  cmd = "%s --port 8604 --model %s --valid_labels %s -logtostderr &" % (
            server, args.crf_model, valid_labels)
  print(cmd)
  print("\nPress [Enter] when Nice2Server has started\n")
  os.system(cmd)

''' SAMPLE COMMAND
python3 py/predict.py \
    --binary examples/stripped/lcrack \ --output ./lcrack.output \ --elf_modifier cpp/modify_elf.so \
    -two_pass \
    --fp_model models/variable/x86/ \
    --n2p_url http://localhost:8604
'''
# Invoke debin on a single file
def debin_predict_file(args, binpath, debpath):
  elf_modifier = os.path.join(args.debin_dir, "cpp/modify_elf.so")
  predict = os.path.join(args.debin_dir, "py/predict.py")
  n2p_url = "http://localhost:8604"
  cmd = "python3 %s --binary %s --output %s --elf_modifier %s -two_pass --fp_model %s --n2p_url %s" % (
            predict, binpath, debpath, elf_modifier, args.var_model, n2p_url)
  print(cmd)
  status = os.system(cmd)
  print("status=%s" % status)

# Invoke debin on an entire directory
def debin_predict_dir(args):
  filenames = os.listdir(args.input_dir)
  path_pairs = map(lambda f : (os.path.join(args.input_dir, f),
                                os.path.join(args.predicted_dir, f)),
                    filenames)
  total = len(filenames)
  for i, (binpath, debpath) in enumerate(path_pairs):
    print("[%s/%s] %s" % (i + 1, total, datetime.datetime.now().time()))
    debin_predict_file(args, binpath, debpath)

# For the directory that debin dumped the modified binaries into, call the dwarf_dumpdir_bickle
def dwarf_dumpdir_bickles(args):
  cmd = "%s %s -py %s -s %s -idir %s -odir %s" % (
            args.python,
            args.dwarf_dumpdir_script,
            args.python,
            args.dwarf_dump_script,
            args.predicted_dir,
            args.output_dir)
  print(cmd)
  status = os.system(cmd)
  print("status=%s" % status)

# Putting everything together
def run(args):
  debin_predict_dir(args)
  dwarf_dumpdir_bickles(args)

def parser():
  parser = argparse.ArgumentParser(description="Mass run debin's predict.py on a bunch of binaries")
  # Have an option for newer python because we would like to use some fancier string formatting
  parser.add_argument("-py", "--python", type=str, default="/usr/bin/python3.7")
  parser.add_argument("-debin", "--debin_dir", type=str, default="/debin")
  parser.add_argument("-crf", "--crf_model", type=str, default="/debin/models/crf/x64/model")
  parser.add_argument("-var", "--var_model", type=str, default="/debin/models/variable/x64")
  parser.add_argument("-dwb", "--dwarf_dump_script", type=str,
                            default="/root/bityr/scripts/dwarf_dump_bickle.py")
  parser.add_argument("-ddwb", "--dwarf_dumpdir_script", type=str,
                            default="/root/bityr/scripts/dwarf_dumpdir_bickles.py")

  parser.add_argument("-idir", "--input_dir", type=str, required=True)
  parser.add_argument("-pdir", "--predicted_dir", type=str, required=True)
  parser.add_argument("-odir", "--output_dir", type=str, required=True)
  return parser


args = parser().parse_args()

# Start n2p
reset()
start_n2p(args)

# Test files
bubble_stripped = "/debin/my-files/cfiles/bubble.stripped"
bubble_debin = "/debin/my-dumps/bubble.debin"

# Need to run the following manually after n2p has started
# debin_predict_dir(args)

