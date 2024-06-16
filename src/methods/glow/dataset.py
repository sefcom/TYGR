import torch
from torch.utils.data import Dataset

def collate_fn(list_samples):
  y_ids_list = []
  y_list = []

  node_labels = []
  edges = []
  edge_labels = []
  node_offset = 0
  var_gather = []
  var_scatter = []
  num_vars = 0
  list_num_vars = []
  for x, y_ids, y in list_samples:
    y_ids_list.append(y_ids)
    y_list.append(y)

    node_labels.append(x.node_labels)
    cur_edges = x.edges + node_offset
    edges.append(cur_edges)
    edge_labels.append(x.edge_labels)

    for nodes in x.var_nodes:
      var_gather += [t + node_offset for t in nodes]
      var_scatter += [num_vars] * len(nodes)
      num_vars += 1
    list_num_vars.append(len(x.var_nodes))
    node_offset += x.node_labels.shape[0]

  node_labels = torch.cat(node_labels, dim=0)
  edge_labels = torch.cat(edge_labels, dim=0)
  edges = torch.cat(edges, dim=1)
  var_gather = torch.LongTensor(var_gather).to(node_labels.device)
  var_scatter = torch.LongTensor(var_scatter).to(var_gather.device)
  y_ids = torch.cat(y_ids_list, dim=0)
  y = torch.cat(y_list, dim=0)

  input_data = (list_num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, y_ids)
  output_data = (y_ids, y)

  return input_data, output_data

class GlowDataset(Dataset):
  def __init__(self, list_samples, method):
    super(GlowDataset, self).__init__()
    self.list_samples = list_samples
    self.method = method

  def __len__(self):
    return len(self.list_samples)

  def __getitem__(self, idx):
    input, output = self.list_samples[idx]
    x = self.method.preproc_input(input, output)
    y_ids = self.method.preproc_output(input, output, 'idx')
    y = self.method.preproc_output(input, output, 'onehot')
    y_ids = y_ids.to(y.device)
    return x, y_ids, y
