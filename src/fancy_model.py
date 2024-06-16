import torch
from torch import nn, Tensor, optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F


class AutoregTypeDecoder(nn.Module):
  def __init__(self, config):
    super(AutoregTypeDecoder, self).__init__()
    self.num_class = config.out_dim
    self.beam_size = config.beam_size
    self.type_embedding = torch.nn.Embedding(self.num_class, config.node_latent_dim)
    self.lstm = nn.LSTM(config.node_latent_dim,
                        config.node_latent_dim,
                        num_layers=3,
                        dropout=config.dropout_rate)
    self.pred_layer = nn.Linear(config.node_latent_dim, self.num_class)

  def get_rnn_init(self, embed):
    _, state = self.lstm(embed)
    return state

  def forward(self, num_vars, x: Tensor, labels: Tensor):
    x_list = x.split(num_vars, dim=0)
    padded_x = pad_sequence(x_list, batch_first=False, padding_value=0.0)
    max_len, bsize = padded_x.shape[0], padded_x.shape[1]
    first_step_state = padded_x[0:1]
    max_steps = max(num_vars)
    if not self.training:
      init_h, init_c = self.get_rnn_init(first_step_state)
      list_scores = []
      list_final_choices = []
      for sample_idx in range(bsize):
        state_decode = first_step_state[:, sample_idx]
        h, c = init_h[:, [sample_idx]], init_c[:, [sample_idx]]
        var_state = padded_x[:, sample_idx]
        prefix_scores = torch.zeros(1).to(state_decode.device)
        choices = torch.LongTensor([[]]).to(state_decode.device)
        for step in range(num_vars[sample_idx]):
          cur_prob = F.log_softmax(self.pred_layer(state_decode), dim=-1)          
          joint_scores = prefix_scores.unsqueeze(1) + cur_prob
          joint_scores = joint_scores.view(-1)
          k = min(self.beam_size, joint_scores.shape[0])
          prefix_scores, idx_selected = torch.topk(joint_scores, k)
          prev_idx = idx_selected // self.num_class
          cls_choice = idx_selected % self.num_class
          if step + 1 < num_vars[sample_idx]:
            prev_h = torch.index_select(h, 1, prev_idx)
            prev_c = torch.index_select(c, 1, prev_idx)

            cur_cls_embed = self.type_embedding(cls_choice)
            var_step_state = var_state[[step + 1]].repeat(cls_choice.shape[0], 1)
            step_update = cur_cls_embed.unsqueeze(0)
            new_s, (h, c) = self.lstm(step_update, (prev_h, prev_c))
            state_decode = new_s[0] + var_step_state
          new_arg_choices = choices[prev_idx]
          new_arg_choices = torch.cat((new_arg_choices, cls_choice.unsqueeze(1)), dim=1)
          choices = new_arg_choices
        list_final_choices.append(choices)
        list_scores.append(prefix_scores)
      return list_final_choices, list_scores
    else:
      label_embed = self.type_embedding(labels)
      label_list = labels.split(num_vars, dim=0)
      padded_labels = pad_sequence(label_list, batch_first=False, padding_value=0)      
      label_embed = label_embed.split(num_vars, dim=0)      
      padded_label_embed = pad_sequence(label_embed, batch_first=False, padding_value=0.0)
      
      lstm_out = first_step_state
      if padded_x.shape[0] > 1:  # more than 1 var to predict
        init_state = self.get_rnn_init(first_step_state)
        var_update = padded_x[1:]
        step_updates = padded_label_embed[:-1]
        lstm_out, _ = self.lstm(step_updates, init_state)
        lstm_out = lstm_out + var_update
        lstm_out = torch.cat([first_step_state, lstm_out], dim=0)
      padded_labels = padded_labels.view(-1)
      raw_logits = self.pred_layer(lstm_out)
      logits = raw_logits.view(padded_labels.shape[0], -1)
      
      # loss = F.cross_entropy(logits, padded_labels, reduction='none')
      logll = F.log_softmax(logits, dim=-1)
      loss = -logll.gather(1, padded_labels.view(-1, 1))
      loss = loss.view(max_len, bsize)
      t_num_vars = torch.LongTensor(num_vars).to(loss.device)
      mask = torch.arange(max_len, dtype=torch.int64, device=loss.device).unsqueeze(1) < t_num_vars.unsqueeze(0)
      loss = torch.mean(torch.sum(loss * mask.float(), dim=0))
      
      active_logits = []
      for i in range(bsize):
        active_logits.append(raw_logits[:num_vars[i], i])
      return torch.cat(active_logits, dim=0), loss


if __name__ == '__main__':
  class Config:
    def __init__(self):
      self.node_latent_dim = 64
      self.out_dim = 3
      self.beam_size = 4

  torch.manual_seed(1)
  config = Config()
  net = AutoregTypeDecoder(config)
  num_vars = [2, 3]
  vec_repr = torch.randn(5, config.node_latent_dim)
  labels = torch.LongTensor([0, 1, 2, 1, 0])

  optimizer = optim.Adam(net.parameters(), lr=1e-4)
  from tqdm import tqdm
  pbar = tqdm(range(2000))
  for _ in pbar:
    optimizer.zero_grad()
    _, loss = net(num_vars, vec_repr, labels)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()
    pbar.set_description('loss: %.4f' % loss.item())
  net.eval()
  choices, scores = net(num_vars, vec_repr, labels)
  print(choices)
  print(scores)
  net.train()
  _, loss = net(num_vars, vec_repr, labels)
  print(loss, (scores[0][0] + scores[1][0]) / 2.0)