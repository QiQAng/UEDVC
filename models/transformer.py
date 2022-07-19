from __future__ import print_function
from __future__ import division

import numpy as np
import json
import pdb
from tqdm import tqdm
import time
import io , sys
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import framework.configbase
import framework.modelbase
import modules.transformer
import modules.concept_predictor
import metrics.evaluation
import metrics.criterion
from torch.nn.parallel import DistributedDataParallel as DDP

DECODER = 'transformer'
CPTM = 'concept_module'

class TransModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super(TransModelConfig, self).__init__()

  def load(self, cfg_file):
    with open(cfg_file) as f:
      data = json.load(f)
    for key, value in data.items():
      if key != 'subcfgs':
        setattr(self, key, value)
    # initialize config objects
    for subname, subcfg_type in self.subcfg_types.items():
      if subname == DECODER:
        self.subcfgs[subname] = modules.transformer.__dict__[subcfg_type]()
      elif subname == CPTM:
        self.subcfgs[subname] = modules.concept_predictor.__dict__[subcfg_type]()
      self.subcfgs[subname].load_from_dict(data['subcfgs'][subname])
      
class TransModel(framework.modelbase.ModelBase):
  def build_submods(self):
    submods = {}
    submods[DECODER] = modules.transformer.Transformer(self.config.subcfgs[DECODER])
    if CPTM in submods:
      submods[CPTM] = modules.concept_predictor.ConceptPredictor(self.config.subcfgs[CPTM])
    return submods

  def build_loss(self):
    xe = metrics.criterion.LabelSmoothingLoss(0.1,self.config.subcfgs[DECODER].vocab,1)
    if not hasattr(self.config, 'rl'):
      rl = metrics.criterion.RewardLoss()
    else:
      rl = metrics.criterion.RewardLoss(self.config.rl)
    cpt = framework.ops.SigmoidCrossEntropyWithLogitsLoss()
    cls = nn.CrossEntropyLoss(reduction='none')
    return (xe, rl, cpt, cls)

  def forward_vc_event(self, batch_data, task):
    trg = batch_data['caption_ids'].cuda()
    trg = trg[:,:max(batch_data['caption_lens'])]
    vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
    vft = vft[:,:max(vft_len)]
    trg_mask = self.create_masks(trg, task)
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    ent = batch_data['event_times'].cuda() 
    if 'concept_ids' in batch_data:
      cpt = batch_data['concept_ids'].cuda()
      cpt_mask = self.create_masks(cpt, task=None)
      hiddens,_ = self.submods[DECODER](vft, cpt, trg, vft_mask, cpt_mask, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
      outputs = nn.LogSoftmax(dim=-1)(hiddens[:,max(vft_len)+25:])
    else:
      hiddens,_ = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
      outputs = nn.LogSoftmax(dim=-1)(hiddens[:,max(vft_len)+self.config.subcfgs[DECODER].max_events_len:])
    output_label = batch_data['output_label'].cuda()
    output_label = output_label[:,:outputs.size(1)]
    ys = output_label.contiguous().view(-1)
    norm = output_label.ne(1).sum().item()
    loss_mlm = self.criterion[0](outputs.view(-1, outputs.size(-1)), ys, norm)
    return loss_mlm

  def forward_mlm_event(self, batch_data, task):
    trg = batch_data['caption_ids'].cuda()
    trg = trg[:,:max(batch_data['caption_lens'])]
    if self.config.subcfgs[DECODER].has_mvm:
      vft = batch_data['video_ft_mvm'].cuda()
    else: 
      vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
    vft = vft[:,:max(vft_len)]
    trg_mask = self.create_masks(trg, task)
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    ent = batch_data['event_times'].cuda()
    if 'concept_ids' in batch_data:
      cpt = batch_data['concept_ids'].cuda()
      cpt_mask = self.create_masks(cpt, task=None)
      hiddens,hiddens_mvm = self.submods[DECODER](vft, cpt, trg, vft_mask, cpt_mask, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
      outputs = nn.LogSoftmax(dim=-1)(hiddens[:,max(vft_len)+25:])
    else:
      hiddens,hiddens_mvm = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
      outputs = nn.LogSoftmax(dim=-1)(hiddens[:,max(vft_len)+self.config.subcfgs[DECODER].max_events_len:])
    output_label = batch_data['output_label'].cuda()
    output_label = output_label[:,:outputs.size(1)]
    ys = output_label.contiguous().view(-1)
    norm = output_label.ne(1).sum().item()
    loss_mlm = self.criterion[0](outputs.view(-1, outputs.size(-1)), ys, norm)
    if self.config.subcfgs[DECODER].has_mvm:
      video_ignore = batch_data['video_ignore'].cuda()
      video_label = batch_data['video_label'].cuda()
      video_ignore = video_ignore[:,:max(vft_len)]
      video_label = video_label[:,:max(vft_len)][video_ignore==1]
      video_pred  = hiddens_mvm[:,:max(vft_len)][video_ignore==1]
      loss_mvm = F.mse_loss(video_pred.view(-1, video_pred.size(-1)), video_label.view(-1, video_label.size(-1)), reduction='none')
      loss_mvm = loss_mvm.mean()
      loss = loss_mlm + loss_mvm*self.config.subcfgs[DECODER].mvm_loss_w
    else:
      loss = loss_mlm
    return loss

  def forward_mem_event(self,batch_data,task):
    vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    vft = vft[:,:max(vft_len)]
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    trg = batch_data['cls'].cuda()
    trg_mask = self.create_masks(trg, task)
    ent = batch_data['event_times'].cuda()
    output_label = batch_data['output_label'].cuda()
    ent_mask = batch_data['ent_masks'].cuda()
    ignore_label = batch_data['ignore'].cuda()
    if task == 'eg':
      ent_mask =self.create_eg_masks(ent_mask)
    else:
      ent_mask = ent_mask.unsqueeze(1)
    if 'video_cls' in batch_data:
      video_cls = batch_data['video_cls'].cuda()
      _,hiddens = self.submods[DECODER](vft, video_cls, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
    else:
      _,hiddens = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
    if task == 'eg' or self.config.subcfgs[DECODER].has_cls == False:
      outputs = hiddens[:,max(vft_len):]
    else:
      outputs = hiddens[:,max(vft_len):-1]
    output_label = output_label[:,:outputs.size(1)]
    outputs = outputs[ignore_label==1]
    output_label = output_label[ignore_label==1]
    if self.config.subcfgs[DECODER].mem_fr:
      loss = F.mse_loss(outputs.view(-1, outputs.size(-1)), output_label.view(-1, output_label.size(-1)), reduction='none')
    else:
      loss = F.binary_cross_entropy_with_logits(outputs.view(-1, outputs.size(-1)), output_label.view(-1, output_label.size(-1)),reduction='none')
    loss = torch.sum(torch.sum(loss, 1))/loss.size(0)
    loss_sum = loss*self.config.subcfgs[DECODER].mem_loss_w 
    return loss_sum

  def forward_sty_event(self,batch_data,task):
    vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    vft = vft[:,:max(vft_len)]
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    hiddens_video = self.submods[DECODER](vft, None, None, vft_mask, None, None, task=task)
    video_cls = batch_data['video_cls'].cuda()
    output_video = hiddens_video[:,max(vft_len)-1]
    loss_video_cls = self.criterion[3](output_video,video_cls).mean()
    return loss_video_cls
    

  def forward_vem_event(self,batch_data,task):
    vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    vft = vft[:,:max(vft_len)]
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    ent = batch_data['event_times'].cuda()
    output_label = batch_data['cls_target'].cuda()
    ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
    trg = batch_data['cls'].cuda()
    trg_mask = self.create_masks(trg, task)
    hiddens = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
    vem_loss = self.criterion[3](hiddens[:,-1], output_label).mean()
    return vem_loss

  def forward_ecm_event(self,batch_data,task):
    trg = batch_data['caption_ids'].cuda()
    trg = trg[:,:max(batch_data['caption_lens'])]
    vft = batch_data['video_ft'].cuda()
    vft_len = batch_data['ft_len'].cuda()
    ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
    vft = vft[:,:max(vft_len)]
    trg_mask = self.create_masks(trg, task)
    vft_mask = self.img_mask(vft_len).unsqueeze(1)
    ent = batch_data['event_times'].cuda()
    output_label = batch_data['cls_target'].cuda()
    hiddens = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
    ecm_loss = self.criterion[3](hiddens[:,max(vft_len)+self.config.subcfgs[DECODER].max_events_len], output_label).mean()
    return ecm_loss

  def forward_loss(self, batch_data, TRG, task='mlm', step=None):

    if task == 'mlm' or task =='vc':
      trg = batch_data['caption_ids'].cuda()
      trg = trg[:,:max(batch_data['caption_lens'])]
      vft = batch_data['video_ft'].cuda()
      vft_len = batch_data['ft_len'].cuda()
      vft = vft[:,:max(vft_len)]
      trg_mask = self.create_masks(trg, task)
      vft_mask = self.img_mask(vft_len).unsqueeze(1)
      if task == 'mlm':
        loss = self.forward_mlm_event(batch_data,task)
      else:
        loss = self.forward_vc_event(batch_data,task)

      return loss
    elif task == 'mem' or task == 'eg':
      return self.forward_mem_event(batch_data,task)
    elif task =='vem':
      return self.forward_vem_event(batch_data,task)
    elif task =='ecm':
      return self.forward_ecm_event(batch_data,task)

  def evaluate(self, tst_reader, do_eval=True,is_tst=False):
    pred_sents, names = [], []
    sty_list = []
    pred_eg_logit = []
    raw_ents,raw_len_ent = [], []
    score = {}
    if isinstance(self.submods[DECODER], DDP):
      Decoder = self.submods[DECODER].module
    else:
      Decoder = self.submods[DECODER]
    for task in tst_reader:
      n_correct, n_word = 0, 0
      tiou_num = 0
      tiou_sum = np.zeros(4)
      tiou_recall = np.zeros(4)
      tiou_percision = np.zeros(4)

      cur_reader = tst_reader[task]
      
      for batch_data in tqdm(cur_reader):
        vft = batch_data['video_ft'].cuda()
        vft_len = batch_data['ft_len'].cuda()
        vft = vft[:,:max(vft_len)]
        vft_mask = self.img_mask(vft_len).unsqueeze(1)
        names.extend(batch_data['names'])
        if task == 'vc':
          if is_tst:
            beam_size = 1
          else:
            beam_size = 1
          ent = batch_data['event_times'].cuda()
          ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
          if 'concept_ids' in batch_data:
            cpt = batch_data['concept_ids'].cuda()
            cpt_mask = self.create_masks(cpt, task=None) 
            output = Decoder.beam_search(vft, ent, vft_mask, ent_mask,cpt,cpt_mask,beam_size=beam_size)
          else:
            output = Decoder.beam_search(vft, ent, vft_mask, ent_mask,beam_size=beam_size)
          captions = cur_reader.dataset.int2sent(output.detach())
          pred_sents.extend(captions)
        
        elif task == 'eg':
          _names = batch_data['names']
          if "video_cls" in batch_data:
            video_cls = batch_data['video_cls'].cuda()  
          else:
            video_cls = None
          output,output_logits = Decoder.pred_lr_ent_ft(vft, vft_mask,video_cls)
          captions,raw_ent,_list_max_ent= cur_reader.dataset.int2eventlr(output.detach(),_names)
          _recall,_percision =metrics.evaluation.compute_tiou_for_eg(captions,raw_ent)
          tiou_num = tiou_num + 1
          tiou_recall = tiou_recall + _recall
          tiou_percision = tiou_percision + _percision
          pred_sents.extend(captions)
          raw_ents.extend(raw_ent)
          raw_len_ent.extend(_list_max_ent)
          if is_tst:
            with torch.cuda.device_of(output_logits):
              output_logits = output_logits.tolist()
              pred_eg_logit.extend(output_logits)

        elif task == 'mlm':
          trg = batch_data['caption_ids'].cuda()
          trg = trg[:,:max(batch_data['caption_lens'])]
          trg_mask = self.create_masks(trg, task)
          output_label = batch_data['output_label'].cuda()
          ent = batch_data['event_times'].cuda()
          ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
          if 'concept_ids' in batch_data:
            cpt = batch_data['concept_ids'].cuda()
            cpt_mask = self.create_masks(cpt, task=None) 
            hidden_mlm,_= self.submods[DECODER](vft, cpt, trg, vft_mask, cpt_mask, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
            output = hidden_mlm[:,max(vft_len)+25:]
          else:
            hidden_mlm,_= self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
            output = hidden_mlm[:,max(vft_len)+self.config.subcfgs[DECODER].max_events_len:] 
          output_label = output_label[:,:output.size(1)]
          output = output[output_label != 1]
          output_label = output_label[output_label != 1]
          n_correct += (output.max(dim=-1)[1] == output_label).sum().item()
          n_word += output_label.numel()

        elif task == 'mem':
          ent = batch_data['event_times'].cuda()
          output_label = batch_data['output_label'].cuda()
          ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
          ignore_label = batch_data['ignore'].cuda()
          trg = batch_data['cls'].cuda()
          trg_mask = self.create_masks(trg, task)
          if self.config.subcfgs[DECODER].has_sty:
            output_video_label = batch_data['video_cls'].cuda()
          else:
            output_video_label  = None
          output,_ = self.submods[DECODER](vft,output_video_label , trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)
          if self.config.subcfgs[DECODER].has_cls:
            output = output[:,max(vft_len):-1]
          else:
            output = output[:,max(vft_len):] 
          output_label = output_label[:,:output.size(1)]
           
          outputs = output[ignore_label==1]
          output_label = output_label[ignore_label==1]
          _iou_sum,_iou_true,mask_true,mask_num = metrics.evaluation.compute_mem_lr(outputs.view(-1, outputs.size(-1)), output_label.view(-1, output_label.size(-1)))
          n_correct += mask_true
          n_word += mask_num
          tiou_sum += _iou_true
          tiou_num += mask_num//2

        elif task == 'vem':
          trg = batch_data['cls'].cuda()
          trg_mask = self.create_masks(trg, task)
          output_label = batch_data['cls_target'].cuda()
          ent = batch_data['event_times'].cuda()
          ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
          output = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)[:,-1]
          n_correct += (output.max(dim=-1)[1] == output_label).sum().item()
          n_word += output_label.numel()
        
        elif task == 'ecm':
          trg = batch_data['caption_ids'].cuda()
          trg = trg[:,:max(batch_data['caption_lens'])]
          trg_mask = self.create_masks(trg, task)
          output_label = batch_data['cls_target'].cuda()
          ent = batch_data['event_times'].cuda()
          ent_mask = batch_data['ent_masks'].cuda().unsqueeze(1)
          output = self.submods[DECODER](vft, None, trg, vft_mask, None, trg_mask, task=task,ent=ent,ent_mask=ent_mask)[:,max(vft_len)+self.config.subcfgs[DECODER].max_events_len]
          n_correct += (output.max(dim=-1)[1] == output_label).sum().item()
          n_word += output_label.numel()

      if task == 'vc':
        if do_eval:
          score.update(metrics.evaluation.compute(pred_sents, cur_reader.dataset.ref_captions, names))
        pred_sents = [{'image_id': name, 'caption': caption} for (name, caption) in zip(names, pred_sents)]
      elif task == 'eg':
        _thres_3_recall,_thres_5_recall,_thres_7_recall,_thres_9_recall = (tiou_recall/tiou_num).tolist()
        _thres_3_per,_thres_5_per,_thres_7_per,_thres_9_per = (tiou_percision/tiou_num).tolist()
        score.update({task+'_acc_iou@0.3_recall':_thres_3_recall})
        score.update({task+'_acc_iou@0.5_recall':_thres_5_recall})
        score.update({task+'_acc_iou@0.7_recall':_thres_7_recall})
        score.update({task+'_acc_iou@0.9_recall':_thres_9_recall})
        score.update({task+'_acc_iou@0.3_per':_thres_3_per})
        score.update({task+'_acc_iou@0.5_per':_thres_5_per})
        score.update({task+'_acc_iou@0.7_per':_thres_7_per})
        score.update({task+'_acc_iou@0.9_per':_thres_9_per})
        pred_sents = [{'image_id': name, 'events': caption,'raw_ents': raw_ent,'len_ft':len_ent} for (name, caption, raw_ent, len_ent) in zip(names, pred_sents,raw_ents,raw_len_ent)]
      elif task == 'mlm' or task=='vem' or task == 'ecm':
        score.update({task+'_avg_acc':n_correct/n_word})
      elif task == 'mem':
        score.update({task+'_avg_acc':n_correct/n_word})
        _thres_3,_thres_5,_thres_7,_thres_9 = (tiou_sum/tiou_num).tolist()
        score.update({task+'_acc_iou@0.3':_thres_3})
        score.update({task+'_acc_iou@0.5':_thres_5})
        score.update({task+'_acc_iou@0.7':_thres_7})
        score.update({task+'_acc_iou@0.9':_thres_9})
    return score, pred_sents

  def validate(self, val_reader):
    self.eval_start()
    metrics, pred_sents = self.evaluate(val_reader)
    return metrics, pred_sents

  def test(self, tst_reader, tst_pred_file, tst_model_file=None, do_eval=True):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()
    metrics, pred_data = self.evaluate(tst_reader, do_eval=do_eval,is_tst=True)
    with open(tst_pred_file, 'w') as f:
      json.dump({"rst":pred_data}, f, indent=1)
    return metrics

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

  def create_masks(self, trg, task=None):
    trg_mask = (trg != 1).unsqueeze(-2)
    if task == 'vc':
      size = trg.size(1) # get seq_len for matrix
      np_mask = self.nopeak_mask(size)
      trg_mask = trg_mask & np_mask
    return trg_mask

  def create_eg_masks(self,ent_mask):
    """
    ent_mask: [batch_size,max_ents_len]
    """
    size = ent_mask.size(1) # get max_ents_len for matrix
    ent_mask = ent_mask.unsqueeze(-2)
    np_mask = self.nopeak_mask(size)
    ent_mask = ent_mask & np_mask
    return ent_mask

  def img_mask(self, lengths, max_len=None):
    ''' Creates a boolean mask from sequence lengths.
        lengths: LongTensor, (batch, )
    '''
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    return ~(torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .ge(lengths.unsqueeze(1)))
