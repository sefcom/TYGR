import argparse
import sys
import os

def parser():
  parser = argparse.ArgumentParser(description="Run the CSmith script")
  parser.add_argument("--csmith_home", type=str, default="/home/celery/foo/csmith/csmith-2.3.0")
  parser.add_argument("-n", "--num_files", type=int, default=10)
  parser.add_argument("-odir", "--output_dir", type=str, required=True)

  parser.add_argument("--generate_files", action="store_true")
  parser.add_argument("--compile_files", action="store_true")
  parser.add_argument("--compile_x64", action="store_true")
  parser.add_argument("--compile_x86", action="store_true")
  parser.add_argument("--compile_arm64", action="store_true")
  parser.add_argument("--all_opts", action="store_true")
  parser.add_argument("--O0", action="store_true")
  parser.add_argument("--O1", action="store_true")
  parser.add_argument("--O2", action="store_true")
  parser.add_argument("--O3", action="store_true")
  return parser

args = parser().parse_args()

def setup_directories(output_dir):
  # Set up the directories if they don't already exist
  src_dir = os.path.join(output_dir, "src")
  x64_dir = os.path.join(output_dir, "x64")
  x86_dir = os.path.join(output_dir, "x86")
  arm64_dir = os.path.join(output_dir, "arm64")

  mkdir_cmd = f"mkdir -p {src_dir} {x64_dir} {x86_dir} {arm64_dir}"
  err = os.system(mkdir_cmd)
  if not err:
    # print("Successfully created subdirs")
    return (src_dir, x64_dir, x86_dir, arm64_dir)
  else:
    print("Failed to create subdirs")
    return None

def generate_source_files(csmith, src_dir, num_files):
  # Generate all the C files
  for i in range(0, num_files):
    filename = f"cs-{i+1}.c"
    outpath = os.path.join(src_dir, filename)

    # First include all the required flags
    flags = [
        "--no-bitfields",
        "--no-compound-assignment",
        "--no-embedded-assigns",
        "--float",
        "--main",
        "--math64",
      ]

    cmd = f"{csmith} -o {outpath}"
    for flag in flags:
      cmd += " " + flag

    print(f"[{i+1}/{num_files}] {cmd}")
    err = os.system(cmd)
    if not err:
      print("Success")
    else:
      print("Failed")

def generate_compiler_options(arch, opt_level=0, csmith_home=args.csmith_home):
  runtime = os.path.join(csmith_home, "runtime")
  if arch == "x64":
    return {
      "compiler" : "gcc",
      "flags" : f"-gdwarf -w -I {runtime} -O{opt_level}"
    }

  elif arch == "x86":
    return {
      "compiler" : "gcc",
      "flags" : f"-gdwarf -w -I {runtime} -O{opt_level} -m32"
    }

  elif arch == "arm64":
    return {
      "compiler" : "aarch64-linux-gnu-gcc",
      "flags" : f"-gdwarf -w -I {runtime} -O{opt_level}"
    }

def compile_source_file(filepath, options, outpath):
  compiler = options["compiler"]
  flags = options["flags"]
  cmd = f"{compiler} {flags} {filepath} -o {outpath}"
  print(f"{cmd}")
  err = os.system(cmd)
  if not err:
    # print(f"Successfully compiled {filepath}")
    pass
  else:
    print(f"Failed to compile {filepath}")

# Run the stuffs

# Make sure all the directories are there
dirs = setup_directories(args.output_dir)
if dirs is None:
  print("Exiting ...")
  exit(1)

print("\n")

(src_dir, x64_dir, x86_dir, arm64_dir) = dirs

# If we want to generate new files via csmith, then do so
if args.generate_files:
  csmith = os.path.join(args.csmith_home, "src", "csmith")
  generate_source_files(csmith, src_dir, args.num_files)

print("\n")

# If we are doing any compiling, then do so
if args.compile_files or args.compile_x64 or args.compile_x86 or args.compile_arm64:
  all_files = list(os.listdir(src_dir))
  total_files = len(all_files)

  # Figure out which optimization levels we want
  opt_levels = []
  if args.all_opts or args.O0:
    opt_levels.append(0)

  if args.all_opts or args.O1:
    opt_levels.append(1)

  if args.all_opts or args.O2:
    opt_levels.append(2)

  if args.all_opts or args.O3:
    opt_levels.append(3)

  for i, f in enumerate(os.listdir(src_dir)):
    print(f"[{i+1}/{total_files}] working on {f}")

    src_path = os.path.join(src_dir, f)

    for opt_level in opt_levels:
      x64_path = os.path.join(x64_dir, f"{f}.x64-O{opt_level}")
      x86_path = os.path.join(x86_dir, f"{f}.x86-O{opt_level}")
      arm64_path = os.path.join(arm64_dir, f"{f}.arm64-O{opt_level}")

      x64_opts = generate_compiler_options("x64", opt_level=opt_level)
      x86_opts = generate_compiler_options("x86", opt_level=opt_level)
      arm64_opts = generate_compiler_options("arm64", opt_level=opt_level)

      if args.compile_files or args.compile_x64:
        compile_source_file(src_path, x64_opts, x64_path)

      if args.compile_files or args.compile_x86:
        compile_source_file(src_path, x86_opts, x86_path)

      if args.compile_files or args.compile_arm64:
        compile_source_file(src_path, arm64_opts, arm64_path)

    print("")


print("Done")

