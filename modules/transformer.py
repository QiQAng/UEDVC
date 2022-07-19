from subprocess import check_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import framework.configbase
from framework.ops import l2norm
import math
import time
import pdb
import numpy as np
from modules.transformer_encoder import Encoder, VISEncoder, CrossEncoder, EntEncoder
from modules.common import gelu,Norm
max_ft_len = 100

class TransformerConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super(TransformerConfig, self).__init__()
    self.vocab = 0
    self.max_words_in_sent = 30
    self.max_events_len = 10
    self.max_ft_len = 60
    self.d_model = 512
    self.d_embed = 4096
    self.n_layers = 4
    self.vis_layers = 1
    self.txt_layers = 1
    self.heads = 8
    self.not_mask_cpt = False
    self.has_mvm = False
    self.dropout = 0.1
    self.d_mode = 3
    self.mem_batch_loop_num = 3
    self.mlm_batch_loop_num = 3
    self.is_overlap_ft = False
    self.mem_loss_w = 1.0
    self.mvm_loss_w = 1.0
    self.e_model = 60
    self.has_cls = False
    self.has_sty = False
    self.mem_fr = False
    self.mem_reverse = False

class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RegionFeatureRegression(nn.Module):
    " for MRM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        #self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
        #                         GELU(),
        #                         Norm(hidden_size))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        #hidden = self.net(input_)
        output = F.linear(input_, self.weight.t(), self.bias)
        return output

    
class Transformer(nn.Module):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.config = config
    self.vis_encoder = VISEncoder(self.config.d_model, self.config.d_embed, self.config.vis_layers, self.config.heads, self.config.dropout, self.config.d_mode)
    self.txt_encoder = Encoder(self.config.vocab, self.config.d_model, self.config.txt_layers, self.config.heads, self.config.dropout, self.config.d_mode)
    self.event_encoder = EntEncoder(self.config.d_model,self.config.txt_layers, self.config.heads,self.config.dropout, self.config.d_mode,self.config.e_model)
    self.cross_encoder = CrossEncoder(self.config.d_model, self.config.n_layers, self.config.heads, self.config.dropout)
    if self.config.d_mode == 4:
      self.cpt_encoder = Encoder(self.config.vocab, self.config.d_model, self.config.txt_layers, self.config.heads, self.config.dropout, self.config.d_mode)
      self.cpt_encoder.embed = self.txt_encoder.embed
      self.cpt_encoder.pe.mode = self.txt_encoder.pe.mode = self.event_encoder.pe.mode = self.vis_encoder.pe.mode
    else:
      self.txt_encoder.pe.mode = self.event_encoder.pe.mode = self.vis_encoder.pe.mode 
    self.logit = nn.Linear(self.config.d_model, self.config.vocab)
    self.sigmoid = nn.Sigmoid()
    if self.config.has_cls:
      self.logit_cls = nn.Linear(self.config.d_model, 2)
    if self.config.has_sty:
      self.logit_video_cls = nn.Linear(self.config.d_model, 3)
    self.logit.weight = self.txt_encoder.embed.embed.weight
    self.dropout = nn.Dropout(self.config.dropout)
    self.mem_fr = self.config.mem_fr
    self.has_mvm = self.config.has_mvm
    if self.has_mvm:
      self.logit_mvm = RegionFeatureRegression(self.config.d_model, self.config.d_embed,self.vis_encoder.embed.weight)
    if self.mem_fr:
      self.logit_mem = RegionFeatureRegression(self.config.d_model, self.config.e_model,self.event_encoder.embed.weight)
    else:
      self.logit_mem = nn.Linear(self.config.d_model, self.config.e_model)
      #self.logit_mem = RegionFeatureRegression(self.config.d_model, self.config.e_model,self.event_encoder.embed.weight)
    self.init_weights()

  def init_weights(self,):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward_mlm_event_model(self,vft,trg,trg_mask,vft_mask,ent,ent_mask):
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    t_outputs = self.txt_encoder(trg, trg_mask, mode=1)
    ent_outputs = self.event_encoder(ent, ent_mask, mode=2)
    input = torch.cat([v_outputs, ent_outputs, t_outputs], dim=1)
    mask = torch.cat([vft_mask,ent_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask)
    if self.has_mvm:
      output_mvm = self.logit_mvm(e_outputs)
    else:
      output_mvm = None
    output = self.logit(e_outputs)
    return output,output_mvm
  
  def forward_mem_event_model(self,vft,trg,trg_mask,vft_mask,ent,ent_mask,video_cls):
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    t_outputs = self.txt_encoder(trg, trg_mask, mode=1)
    ent_outputs = self.event_encoder(ent, ent_mask, video_cls, mode=2)
    input = torch.cat([v_outputs, ent_outputs, t_outputs], dim=1)
    mask = torch.cat([vft_mask,ent_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask)
    logits = self.logit_mem(e_outputs)
    if self.mem_fr:
      output = logits
    else:
      output = self.sigmoid(logits)
    return output,logits
  
  def forward_eg_event_model(self,vft,vft_mask,ent,ent_mask,video_cls):
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    ent_outputs = self.event_encoder(ent, ent_mask,video_cls, mode=2)
    input = torch.cat([v_outputs, ent_outputs], dim=1)
    if ent_mask.size(1) != 1:
      firmask = torch.cat([vft_mask, ent_mask[:,0].unsqueeze(1)], dim=-1)
      firmask = firmask.repeat(1, vft.size(1), 1)
      vft_mask = vft_mask.repeat(1, ent.size(1), 1)
      secmask = torch.cat([vft_mask, ent_mask], dim=-1)
      mask = torch.cat([firmask, secmask], dim=1)
    else:
      mask = torch.cat([vft_mask,ent_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask)
    logits = self.logit_mem(e_outputs)
    if self.mem_fr:
      output = logits
    else:
      output = self.sigmoid(logits)
    return output,logits

  def forward_vc_event_model(self,vft,trg,trg_mask,vft_mask,ent,ent_mask,cpt=None,cpt_mask = None):
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    t_outputs = self.txt_encoder(trg, trg_mask, mode=1)
    ent_outputs = self.event_encoder(ent, ent_mask, mode=2)
    if cpt is not None:
      c_outputs = self.cpt_encoder(cpt,cpt_mask,mode=3)
      input = torch.cat([v_outputs, ent_outputs, c_outputs, t_outputs], dim=1)
      if trg_mask is not None and trg_mask.size(1) != 1:
        firmask = torch.cat([vft_mask, ent_mask, cpt_mask, trg_mask[:,0].unsqueeze(1)], dim=-1)
        firmask = firmask.repeat(1, vft.size(1)+ent.size(1)+cpt.size(1), 1)
        vft_mask = vft_mask.repeat(1, trg.size(1), 1)
        ent_mask = ent_mask.repeat(1, trg.size(1), 1)
        cpt_mask = cpt_mask.repeat(1, trg.size(1), 1)
        secmask = torch.cat([vft_mask, ent_mask, cpt_mask, trg_mask], dim=-1)
        mask = torch.cat([firmask, secmask], dim=1)
      else:
        mask = torch.cat([vft_mask,ent_mask, cpt_mask, trg_mask], dim=-1)
    else:
      input = torch.cat([v_outputs, ent_outputs, t_outputs], dim=1)
      if trg_mask is not None and trg_mask.size(1) != 1:
        firmask = torch.cat([vft_mask, ent_mask, trg_mask[:,0].unsqueeze(1)], dim=-1)
        firmask = firmask.repeat(1, vft.size(1)+ent.size(1), 1)
        vft_mask = vft_mask.repeat(1, trg.size(1), 1)
        ent_mask = ent_mask.repeat(1, trg.size(1), 1)
        secmask = torch.cat([vft_mask, ent_mask, trg_mask], dim=-1)
        mask = torch.cat([firmask, secmask], dim=1)
      else:
        mask = torch.cat([vft_mask,ent_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask)
    output = self.logit(e_outputs)
    if self.has_mvm:
      output_mvm = self.logit_mvm(e_outputs)
    else:
      output_mvm = None
    return output,output_mvm

  def forward(self, vft, video_cls, trg, vft_mask, cpt_mask, trg_mask, task='mlm',ent=None,ent_mask=None,vft_len=None):
    if task == 'mlm':
      return self.forward_vc_event_model(vft,trg,trg_mask,vft_mask,ent,ent_mask,video_cls,cpt_mask)
    elif task == 'mem':
      if self.config.has_cls:
        return self.forward_mem_event_model(vft,trg,trg_mask,vft_mask,ent,ent_mask,video_cls)
      else:
        return self.forward_eg_event_model(vft,vft_mask,ent,ent_mask,video_cls)
    elif task == 'vc':
      return self.forward_vc_event_model(vft,trg,trg_mask,vft_mask,ent,ent_mask,video_cls,cpt_mask)
    elif task == 'eg':
      return self.forward_eg_event_model(vft,vft_mask,ent,ent_mask,video_cls)
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    t_outputs = self.txt_encoder(trg, trg_mask, mode=1)
    cpt = None
    if cpt is not None:
      c_outputs = self.cpt_encoder(cpt, cpt_mask, mode=2)
      input = torch.cat([v_outputs, c_outputs, t_outputs], dim=1)
    else:
      input = torch.cat([v_outputs, t_outputs], dim=1)

    if trg_mask is not None and trg_mask.size(1) != 1:
      if cpt is not None:
        firmask = torch.cat([vft_mask, cpt_mask, trg_mask[:,0].unsqueeze(1)], dim=-1)
        firmask = firmask.repeat(1, vft.size(1)+cpt.size(1), 1)
        vft_mask = vft_mask.repeat(1, trg.size(1), 1)
        cpt_mask = cpt_mask.repeat(1, trg.size(1), 1)
        secmask = torch.cat([vft_mask, cpt_mask, trg_mask], dim=-1)
        
      else:
        firmask = torch.cat([vft_mask, trg_mask[:,0].unsqueeze(1)], dim=-1)
        firmask = firmask.repeat(1, vft.size(1), 1)
        vft_mask = vft_mask.repeat(1, trg.size(1), 1)
        secmask = torch.cat([vft_mask, trg_mask], dim=-1)
      mask = torch.cat([firmask, secmask], dim=1)
    else:
      if cpt is not None:
        mask = torch.cat([vft_mask, cpt_mask, trg_mask], dim=-1)
      else:
        mask = torch.cat([vft_mask, trg_mask], dim=-1)

    e_outputs = self.cross_encoder(input, mask)
    output = self.logit(e_outputs)
    return output

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

  def init_vars_mlm(self,vft,ent,vft_mask,ent_mask, cpt=None,cpt_mask = None,beam_size=5):
    # encoder process,init_tok=BOS,
    init_tok, mask_tok = 2, 4
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    ent_outputs = self.event_encoder(ent, ent_mask, mode=2)
    outputs = torch.LongTensor([[init_tok]]*len(vft)).cuda()
    trg_mask = self.nopeak_mask(1).repeat(vft.size(0),1,1)
    t_outputs = self.txt_encoder(outputs, trg_mask, mode=1)
    if cpt is not None:
      c_outputs = self.cpt_encoder(cpt,cpt_mask,mode=3)
      input = torch.cat([v_outputs,ent_outputs,c_outputs,t_outputs], dim=1)
      mask = torch.cat([vft_mask, ent_mask, cpt_mask, trg_mask], dim=-1)
    else:
      input = torch.cat([v_outputs,ent_outputs,t_outputs], dim=1)
      mask = torch.cat([vft_mask, ent_mask, trg_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask, step=1)
    mask_word = torch.ones(vft.size(0), 1).fill_(mask_tok).long().cuda()
    outputs = torch.cat([outputs, mask_word], dim=1)
    trg_mask = self.nopeak_mask(2).repeat(vft.size(0),1,1)
    t_outputs = self.txt_encoder(outputs, trg_mask, mode=1)
    out = self.logit(self.cross_encoder(t_outputs, trg_mask, step=2))
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(beam_size)
    log_scores = torch.log(probs)
    outputs = torch.zeros(len(vft), beam_size, self.config.max_words_in_sent).long().cuda()
    outputs[:, :, 0] = init_tok
    outputs[:, :, 1] = ix
    return outputs, log_scores

  def k_best_outputs(self, outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1).cuda() + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs
    return outputs, log_scores


  def beam_search(self, vft, ent, vft_mask, ent_mask, cpt=None, cpt_mask=None, beam_size=5):
    outputs, log_scores = self.init_vars_mlm(vft, ent, vft_mask, ent_mask, cpt, cpt_mask, beam_size)
    eos_tok, mask_tok = 3, 4
    final = torch.zeros(len(vft), self.config.max_words_in_sent).long().cuda()
    mask_word = torch.ones(1, 1).fill_(mask_tok).long().cuda()
    for i in range(2, self.config.max_words_in_sent):
      tmp = outputs.view(-1,outputs.size(-1))[:,:i]
      tmp = torch.cat([tmp, mask_word.repeat(tmp.size(0),1)], dim=1)
      trg_mask = self.nopeak_mask(i+1).repeat(tmp.size(0),1,1)
      t_outputs = self.txt_encoder(tmp, trg_mask, mode=1)
      out = self.logit(self.cross_encoder(t_outputs, trg_mask, step=i+1))
      out = F.softmax(out, dim=-1)
      
      out = out.view(len(vft), beam_size, -1, out.size(-1))
      for b in range(len(vft)):
        outputs[b], log_scores[b] = self.k_best_outputs(outputs[b], out[b], log_scores[b].unsqueeze(0), i, beam_size) 
        ones = (outputs[b]==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs[b]), dtype=torch.long).cuda()
        for vec in ones:
          if sentence_lengths[vec[0]]==0: # First end symbol has not been found yet
            sentence_lengths[vec[0]] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        if num_finished_sentences == beam_size:
          alpha = 1
          div = 1/(sentence_lengths.type_as(log_scores[b])**alpha)
          _, ind = torch.max(log_scores[b] * div, 0)
          if final[b].sum() == 0:
            final[b] = outputs[b][ind]
    for b in range(len(vft)):
      if final[b].sum() == 0:
        final[b] = outputs[b][0]
    return final

  def pred_lr_ent_ft(self,vft,vft_mask,video_pred_cls=None):
    """masked decoder"""
    final = torch.zeros(len(vft), self.config.max_events_len-1,self.config.max_ft_len).long().cuda()
    final_logits = torch.zeros(len(vft), self.config.max_events_len-1,self.config.max_ft_len).float().cuda()
    default_ft = np.zeros(self.config.max_ft_len)
    end_ent = init_ent = mask_ent = default_ft
    v_outputs = self.vis_encoder(vft, vft_mask, mode=0)
    # outputs: [batch_size,ent_lens,512]
    outputs = torch.FloatTensor([[init_ent]]*len(vft)).cuda()
    ent_mask = self.nopeak_mask(1).repeat(vft.size(0),1,1)
    ent_outputs = self.event_encoder(outputs, ent_mask,video_pred_cls, mode=2)
    input = torch.cat([v_outputs,ent_outputs], dim=1)
    mask = torch.cat([vft_mask, ent_mask], dim=-1)
    e_outputs = self.cross_encoder(input, mask, step=1)
    mask_word = torch.FloatTensor([[mask_ent]]*len(vft)).cuda()
    outputs = torch.cat([outputs, mask_word], dim=1)
    ent_mask = self.nopeak_mask(2).repeat(vft.size(0),1,1)
    ent_outputs = self.event_encoder(outputs, ent_mask, video_pred_cls,mode=2)
    out = self.logit_mem(self.cross_encoder(ent_outputs, ent_mask, step=2))
    if not self.mem_fr:
      out = self.sigmoid(out)
    pred_ix = (out[:, -1] > 0.5)
    process_ix = out[:,-1].data.cpu().numpy()
    for batch_id, e_ent  in enumerate(process_ix):
      idx_1_num = np.where(e_ent>0.5)[0]
      if len(idx_1_num) > 0:
        start = idx_1_num[0]
        end = idx_1_num[-1]
        temp = np.zeros(self.config.max_ft_len)
        temp[start:end+1] = 1
        process_ix[batch_id] = temp
      else:
        process_ix[batch_id] = np.zeros(self.config.max_ft_len)
    pred_ix = torch.FloatTensor(process_ix).cuda()
    outputs = torch.zeros(len(vft),self.config.max_events_len,self.config.max_ft_len).float().cuda()
    outputs[:, 0] = torch.FloatTensor(init_ent)
    outputs[:, 1] = pred_ix
    final[:,0] = pred_ix
    final_logits[:,0] = out[:, -1]
    temp_max_pred = torch.zeros(len(vft),self.config.max_ft_len).float().cuda()
    temp_max_pred += pred_ix 
    mask_word =torch.FloatTensor([[mask_ent]]*len(vft)).cuda()
    for i in range(2, self.config.max_events_len):
      tmp = outputs[:,:i]
      tmp = torch.cat([tmp, mask_word], dim=1)
      ent_mask = self.nopeak_mask(i+1).repeat(tmp.size(0),1,1)
      ent_outputs = self.event_encoder(tmp, ent_mask, video_pred_cls,mode=2)
      out = self.logit_mem(self.cross_encoder(ent_outputs, ent_mask, step=i+1))
      if not self.mem_fr:
        out = self.sigmoid(out)
      pred_ix = (out[:, -1] > 0.5)
      process_ix = out[:,-1].data.cpu().numpy()
      for batch_id, e_ent  in enumerate(process_ix):
        idx_1_num = np.where(e_ent>0.5)[0]
        if len(idx_1_num) > 0:
          start = idx_1_num[0]
          end = idx_1_num[-1]
          temp = np.zeros(self.config.max_ft_len)
          temp[start:end+1] = 1
          process_ix[batch_id] = temp
        else:
          process_ix[batch_id] = np.zeros(self.config.max_ft_len)
      pred_ix = torch.FloatTensor(process_ix).cuda()
      temp_max_pred += pred_ix
      if self.config.is_overlap_ft:
        outputs[:, i] = temp_max_pred.masked_fill(pred_ix==0,0)
      else:
        outputs[:, i] = pred_ix
      final[:,i-1] = pred_ix
      final_logits[:,i-1] = out[:, -1] 
    return final, final_logits
