import os
import torch

def setup_cuda(gpu):
  # print("=======================")
  # print(gpu)
  if gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
