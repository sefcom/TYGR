from typing import *
from torch import Tensor

from . import preproc


def accuracy(config: preproc.Config, y_pred: Tensor, y: Tensor) -> Tuple[int, int]:
  (n, _) = y_pred.size()

  tP = 0
  fP = 0
  fN = 0
  acc = 0

  result = []
  allType = []
  total_count = 0

  ###  get # of different type and all result
  for i in range(n):
    ty = config.type_set.tensor_to_type(y[i])
    ty_pred = config.type_set.tensor_to_type(y_pred[i])
    
    total_count += 1
    if ty == ty_pred:
      acc += 1

    if ty not in allType:
      allType.append(ty)
    result.append([ty,ty_pred])

  for ty in allType:
    for predRes in result:
      if ty in predRes:
          # actual type & predicted type
          if ty == predRes[0] and ty == predRes[1]:
              tP += 1
          elif ty == predRes[0] and ty != predRes[1]:
              fN += 1
          elif ty != predRes[0] and ty == predRes[1]:
              fP += 1

  return (acc, tP, fN, fP, total_count)


# def accuracy(config: preproc.Config, y_pred: Tensor, y: Tensor) -> Tuple[int, int]:
#   (n, _) = y_pred.size()
#   total_count = 0
#   accurate_count = 0
#   for i in range(n):
#     ty_pred = config.type_set.tensor_to_type(y_pred[i])
#     ty = config.type_set.tensor_to_type(y[i])
#     total_count += 1
#     if ty_pred == ty:
#       accurate_count += 1
#   return (accurate_count, total_count)

def topk_accuracy(config: preproc.Config, k: int, y_pred: Tensor, y: Tensor) -> Tuple[int, int]:
  (n, _) = y_pred.size()
  total_count = 0
  accurate_count = 0
  for i in range(n):
    ty_preds = config.type_set.tensor_to_topk_types(y_pred[i], k)
    ty = config.type_set.tensor_to_type(y[i])
    total_count += 1
    if ty in ty_preds:
      accurate_count += 1
  return (accurate_count, total_count)
