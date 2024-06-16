from src.analysis.angr.ast_graph import Label, NodeLabel
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, MessagePassing, global_max_pool
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax
from src.fancy_model import AutoregTypeDecoder
import numpy as np

def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return orig_smax(src, index, dim, out, dim_size)[0]

from .common import PreprocGlowInput
from . import preproc

class Config:
  def __init__(
    self,
    preproc_config: preproc.Config,

    # Node label encoder
    node_encoder_num_layers: int = 2,
    node_latent_dim: int = 128,

    # GNN
    num_msg_pass_layers: int = 8,
    num_decoder_layers: int = 2,
    share_weight: bool = False,
    activation: str = "relu",
    gnn_type: str = 'rgcn',
    aggregation: str = 'add',
    node_aggregation: str = 'mean',
    decoder_type: str = 'autoreg',
    beam_size: int = 4,
    dropout_rate: float = 0,
  ):
    self.preproc_config = preproc_config

    self.node_encoder_in_dim = self.preproc_config.node_label_encoding_dim
    self.node_encoder_num_layers = node_encoder_num_layers
    self.node_latent_dim = node_latent_dim

    #self.edge_dim = self.preproc_config.num_edge_ops
    self.edge_dim = 44
    self.num_msg_pass_layers = num_msg_pass_layers
    self.share_weight = share_weight
    self.activation = activation
    self.gnn_type = gnn_type
    self.aggregation = aggregation
    self.node_aggregation = node_aggregation
    self.num_decoder_layers = num_decoder_layers
    self.decoder_type = decoder_type
    self.beam_size = beam_size
    self.out_dim = self.preproc_config.type_set.num_types()
    print("hello")
    print(self.preproc_config.type_set.num_types())
    self.dropout_rate = dropout_rate

class NodeLabelEncoder(nn.Module):
  def __init__(self, config: Config):
    super().__init__()

    # First layer
    self.first_layer = nn.Sequential(
      nn.Linear(config.node_encoder_in_dim, config.node_latent_dim),
    )

    # Layers
    layers = []
    for _ in range(config.node_encoder_num_layers - 1):
      layers.append(nn.ReLU())
      layers.append(nn.Linear(config.node_latent_dim, config.node_latent_dim))
    self.layers = nn.Sequential(*layers)

  def forward(self, node_label: Tensor):
    y = self.first_layer(node_label)
    if len(self.layers) > 1:
      y = self.layers(y)
    return y

class TypeDecoder(nn.Module):
  def __init__(self, config: Config):
    super().__init__()

    # Layers
    layers = []
    for _ in range(config.num_decoder_layers - 1):
      layers.append(nn.Linear(config.node_latent_dim, config.node_latent_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(p=config.dropout_rate))
    self.layers = nn.Sequential(*layers)

    # Last layer
    self.last_layer = nn.Linear(config.node_latent_dim, config.out_dim)

  def forward(self, num_vars, x: Tensor, labels: Tensor):
    y = self.layers(x)
    y = self.last_layer(y)
    return y, None

class GGNNConv(MessagePassing):
  def __init__(self, node_state_dim: int, num_edge_types: int, aggr='add'):
    super(GGNNConv, self).__init__(aggr=aggr)
    self.num_edge_types = num_edge_types
    self.edge_embedding = torch.nn.Embedding(num_edge_types, node_state_dim)
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(3 * node_state_dim, 2 * node_state_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(2 * node_state_dim, node_state_dim)
    )
    torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
    if aggr == 'sum':
      aggr = 'add'

  def forward(self, x, edge_index, edge_labels):
    edge_embeddings = self.edge_embedding(edge_labels)
    return self.propagate(edge_index, x=x, edge_features=edge_embeddings)

  def message(self, x_i, x_j, edge_features):
    edge_inputs = torch.cat((x_i, x_j, edge_features), dim=-1)
    return self.mlp(edge_inputs)

# aggregator
def get_aggr(config):
  if config.node_aggregation == 'add':
    readout_agg = scatter_add
  elif config.node_aggregation == 'mean':
    readout_agg = scatter_mean
  elif config.node_aggregation == 'max':
    readout_agg = scatter_max
  else:
    raise ValueError('unknown aggr type %s' % config.node_aggregation)
  return readout_agg


class GlowGNN(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self.activation = nn.GELU()

    # Encoder
    self.encoder = NodeLabelEncoder(config)

    # GNNs
    self.rgcns = nn.ModuleList()
    if self.config.gnn_type == 'rgcn':
      layer_fn = lambda: RGCNConv(
        self.config.node_latent_dim,
        self.config.node_latent_dim,
        self.config.edge_dim,
        num_bases=10,
        bias=True)
    elif self.config.gnn_type == 'ggnn':
      layer_fn = lambda: GGNNConv(
        self.config.node_latent_dim,
        self.config.edge_dim,
        aggr=self.config.aggregation,
      )
    else:
      raise ValueError('unknown gnn type %s' % self.config.gnn_type)

    # Share weight
    if self.config.share_weight:
      layer = layer_fn()
      for _ in range(self.config.num_msg_pass_layers):
        self.rgcns.append(layer)
    else:
      for _ in range(self.config.num_msg_pass_layers):
        self.rgcns.append(layer_fn())

    # Decoder
    if config.decoder_type == 'independent':
      self.decoder = TypeDecoder(config)
    elif config.decoder_type == 'autoreg':
      self.decoder = AutoregTypeDecoder(config)
    else:
      raise NotImplementedError

  def forward(self, x) -> Tensor:
    (num_vars, node_labels, edge_labels, edges, var_gather, var_scatter, labels) = x
    node_labels = self.encoder(node_labels)

    # confirm layer dimension
    #########
    edge_dim = edge_labels.cpu().numpy()

    for layer in self.rgcns:
      node_labels = layer(node_labels, edges, edge_labels)
      node_labels = self.activation(node_labels)
      node_labels = F.dropout(node_labels, p=self.config.dropout_rate, training=self.training)

    node_repr = node_labels[var_gather]
    readout_agg = get_aggr(self.config)
    var_repr = readout_agg(node_repr, var_scatter, dim=0, dim_size=sum(num_vars))
    return self.decoder(num_vars, var_repr, labels)