import sys
import pickle


def _displayhook(o):
  if type(o).__name__ in ('int', 'long'):
    print(hex(o))
    __builtins__._ = o
  else:
    sys.__displayhook__(o)

sys.displayhook = _displayhook

data = pickle.load(open(sys.argv[1], "rb"))

for addr, rhs in data.items():
  print(f"{hex(addr)} ---")
  for loc, btys in rhs.items():
    print(f"\t{loc}")
    for (r0, r1, bty) in btys:
      if isinstance(r0, int) and isinstance(r1, int):
        print(f"\t\t({hex(r0)}, {hex(r1)})\t{bty}")
      elif isinstance(r0, int) and r1 == None:
        print(f"\t\t({hex(r0)}, {'_' * len(hex(r0))})\t{bty}")

      elif r0 == "pc" and isinstance(r1, int):
        print(f"\t\tpc @ {hex(r1)}\t{bty}")

      else:
        print(f"Malformed ranges {r0}, {r1}")


