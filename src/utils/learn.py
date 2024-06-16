from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class Metric:
  def __init__(
    self,
    accuracy: float,
    topk_accuracy: float,
    loss: float,
    k: int,
  ):
    self.accuracy = accuracy
    self.topk_accuracy = topk_accuracy
    self.loss = loss
    self.k = k

  def print(self):
    print(f"Accuracy: {self.accuracy}")
    print(f"Top-{self.k} Accuracy: {self.accuracy}")
    print(f"Loss: {self.loss}")

def run(
  method,
  model,
  dataset,
  args,
  prompt="[]",
  progress_bar = True,
  optimizer = None,
  callback = None,
  shuffle = False,
) -> Metric:
  # Stats
  total_loss = 0
  num_processed = 0
  accurate_count = 0
  total_count = 0
  topk_accurate_count = 0
  topk_total_count = 0
  tPositive = 0
  fNegative = 0
  fPositive = 0

  dataset = method.dataset(dataset)
  data_loader = DataLoader(dataset,
                           batch_size=args.batch_size,
                           shuffle=shuffle,
                           collate_fn=method.collate_function(),
                           drop_last=True)

  # Iteration
  iterator = tqdm(data_loader) if progress_bar else data_loader
  for (x, y) in iterator:
    y_pred, loss = model(x)

    if loss is None:
      # Loss
      l = method.loss(y_pred, y)
    else:
      l = loss

    # Back prop
    if optimizer is not None:
      optimizer.zero_grad()
      l.backward()
      optimizer.step()

    if loss is None or optimizer is not None:
      # Normal accuracy
      # accurate, total = method.accuracy(y_pred, y)
      # accurate_count += accurate
      # total_count += total

      # # TopK accuracy
      # topk_accurate, topk_total = method.topk_accuracy(args.k, y_pred, y)
      # topk_accurate_count += topk_accurate
      # topk_total_count += topk_total

      # tP, fN, fP, total
      acc, tP, fN, fP, total = method.accuracy(y_pred, y)
      total_count += total
      tPositive += tP
      fNegative += fN
      fPositive += fP
      accurate_count += acc

      # Callback
      if callback is not None:
        callback(y_pred, y)
    else:
      offset = 0
      y_ids, _ = y
      y_ids = y_ids.data.cpu().numpy()
      l = 0.0
      for idx, pred_struct in enumerate(y_pred):
        pred_struct = pred_struct.data.cpu().numpy()

        for i in range(pred_struct.shape[1]):
          hit = -1
          for k in range(0, args.k):
            if k >= pred_struct.shape[0]:
              break
            if pred_struct[k, i] == y_ids[i + offset]:
              hit = k
              break
          accurate_count += hit == 0
          topk_accurate_count += hit >= 0
        offset += pred_struct.shape[1]
        l = l + -loss[idx][0]
      l = l / len(y_pred)
      assert offset == y_ids.shape[0]
      total_count += offset
      topk_total_count += offset
    # Stats
    total_loss += l.item()
    num_processed += 1

    # Print
    if progress_bar:
      curr_loss = total_loss / num_processed

      if(tPositive + fPositive) == 0:
        cur_precision = 0
      else:
        cur_precision = tPositive/(tPositive + fPositive)

      cur_recall = tPositive/(tPositive + fNegative)

      if(cur_precision + cur_recall) == 0:
        cur_f1 = 0
      else:
        cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall)
      accuracy = accurate_count/total_count
      iterator.set_description(f"{prompt}: Avg loss {curr_loss:.4f}, Accuracy{accuracy:.4f}, Precision {cur_precision:.4f},  Recall {cur_recall:.4f},  F1 {cur_f1:.4f}")

  # Overall stats
  # accuracy = accurate_count / total_count
  # topk_accuracy = topk_accurate_count / topk_total_count
  
  precision = tPositive/(tPositive + fPositive)
  recall= tPositive/(tPositive + fNegative)
  loss = total_loss / num_processed
  return Metric(precision, recall, loss, args.k)

def run_fine_tune(
  method,
  model,
  dataset,
  args,
  prompt = "[]",
  progress_bar = True,
  optimizer = None,
  callback = None,
  shuffle = False,
) -> Metric:
  # Stats
  total_loss = 0
  num_processed = 0
  accurate_count = 0
  total_count = 0
  topk_accurate_count = 0
  topk_total_count = 0
  tPositive = 0
  fNegative = 0
  fPositive = 0

  dataset = method.fine_tune_dataset(dataset)
  data_loader = DataLoader(dataset,
                           batch_size=args.batch_size,
                           shuffle=shuffle,
                           collate_fn=method.collate_function(),
                           drop_last=True)

  # Iteration
  iterator = tqdm(data_loader) if progress_bar else data_loader
  for (x, y) in iterator:
    y_pred, loss = model(x)

    if loss is None:
      # Loss
      l = method.loss(y_pred, y)
    else:
      l = loss

    # Back prop
    if optimizer is not None:
      optimizer.zero_grad()
      l.backward()
      optimizer.step()

    if loss is None or optimizer is not None:
      # Normal accuracy
      # accurate, total = method.accuracy(y_pred, y)
      # accurate_count += accurate
      # total_count += total

      # # TopK accuracy
      # topk_accurate, topk_total = method.topk_accuracy(args.k, y_pred, y)
      # topk_accurate_count += topk_accurate
      # topk_total_count += topk_total

      # tP, fN, fP, total
      acc, tP, fN, fP, total = method.accuracy(y_pred, y)
      total_count += total
      tPositive += tP
      fNegative += fN
      fPositive += fP
      accurate_count += acc

      # Callback
      if callback is not None:
        callback(y_pred, y)
    else:
      offset = 0
      y_ids, _ = y
      y_ids = y_ids.data.cpu().numpy()
      l = 0.0
      for idx, pred_struct in enumerate(y_pred):
        pred_struct = pred_struct.data.cpu().numpy()

        for i in range(pred_struct.shape[1]):
          hit = -1
          for k in range(0, args.k):
            if k >= pred_struct.shape[0]:
              break
            if pred_struct[k, i] == y_ids[i + offset]:
              hit = k
              break
          accurate_count += hit == 0
          topk_accurate_count += hit >= 0
        offset += pred_struct.shape[1]
        l = l + -loss[idx][0]
      l = l / len(y_pred)
      assert offset == y_ids.shape[0]
      total_count += offset
      topk_total_count += offset
    # Stats
    total_loss += l.item()
    num_processed += 1

    # Print
    if progress_bar:
      curr_loss = total_loss / num_processed

      if(tPositive + fPositive) == 0:
        cur_precision = 0
      else:
        cur_precision = tPositive/(tPositive + fPositive)

      cur_recall = tPositive/(tPositive + fNegative)

      if(cur_precision + cur_recall) == 0:
        cur_f1 = 0
      else:
        cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall)
      accuracy = accurate_count/total_count
      iterator.set_description(f"{prompt}: Avg loss {curr_loss:.4f}, Accuracy{accuracy:.4f}, Precision {cur_precision:.4f},  Recall {cur_recall:.4f},  F1 {cur_f1:.4f}")

  # Overall stats
  # accuracy = accurate_count / total_count
  # topk_accuracy = topk_accurate_count / topk_total_count
  precision = tPositive/(tPositive + fPositive)
  recall= tPositive/(tPositive + fNegative)
  loss = total_loss / num_processed
  return Metric(precision, recall, loss, args.k)
