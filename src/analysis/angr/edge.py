ALL_OPS = [
  'mem_data', 'mem_loc',
  'reg_data', 'reg_loc',
  'Concat', 'Extract',
  '__sub__', '__add__', '__mul__', '__floordiv__', 'SDiv',
  '__ge__', 'SLE', '__lt__', 'ULE',
  'SignExt', 'ZeroExt',
  'LShR', '__lshift__', '__rshift__',
  'If', 'Or', 'Not', 'And', '__and__', '__or__', '__xor__', '__eq__', '__invert__',
  'fpToFP', 'fpToUBV', 'fpToIEEEBV', 'fpToSBV', 'fpDiv', 'fpMul', 'fpNeg', 'fpEQ', 'fpAdd', 'fpLT'
]

REDUCED_OPS = [
  'mem_data', 'mem_loc',
  'reg_data', 'reg_loc',
  'Concat', 'Extract',
  '__sub__', '__add__', '__mul__', '__floordiv__', 'SDiv',
  '__ge__', 'SLE', '__lt__', 'ULE',
  'SignExt', 'ZeroExt',
  'LShR', '__lshift__', '__rshift__',
  'If', 'Or', 'Not', 'And', 'Xor', 'Eq',
  'fp_conv', 'fp_op', 'fp_cmp',
]

REDUCE_OP_MAP = {
  "fpToIEEEBV": "fp_conv",
  "fpToFP": "fp_conv",
  'fpToUBV': "fp_conv",
  'fpToSBV': "fp_conv",
  'fpDiv': "fp_op",
  'fpMul': "fp_op",
  'fpNeg': "fp_op",
  'fpAdd': "fp_op",
  'fpEQ': "fp_cmp",
  'fpLEQ': "fp_cmp",
  'fpGEQ': "fp_cmp",
  'fpLT': "fp_cmp",
  'fpGT': "fp_cmp",
  "__or__": "Or",
  "__and__": "And",
  "__invert__": "Not",
  "__eq__": "Eq",
  "__xor__": "Xor",
}
