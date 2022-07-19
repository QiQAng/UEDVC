from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import pdb
import math
import h5py

from cytoolz import partition_all
import torch.utils.data
from torch.utils.data import Sampler
from scipts.rule_for_ent_type import judge_iou_type
UNK, PAD, BOS, EOS, MASK = 0, 1, 2, 3, 4
CLS = 5

class PretrainVCDataset(torch.utils.data.Dataset):
  def __init__(self, name_files, anno_files, concepts, ft_roots, word2int, int2word, 
    max_words_in_sent=30, max_ft_len=50, is_overlap_ft = False, sty_file=None,mem_reverse = False,is_train=False, task='vc', _logger=None, 
    is_generate_only=False,is_use_eventsegment=True,max_events_len=10):
    super(PretrainVCDataset, self).__init__()
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.names, self.name2dataset = [], []
    self.is_generate_only = is_generate_only
    for dataset, name_file in name_files:
      current_data = np.load(name_file)
      self.names.extend(current_data)
      self.name2dataset.extend([dataset] * len(current_data))

    self.ref_captions = {}
    for anno in anno_files:
      self.ref_captions.update(json.load(open(anno)))
    self.num_ft = len(self.names)
    self.print_fn('clip size %d' % self.num_ft)

    self.use_cpt = concepts is not None
    
    if self.use_cpt:
      self.concepts = {}
      for cpt in concepts:
        self.concepts.update(json.load(open(cpt)))
      self.print_fn('concepts size %d' % len(self.concepts))

    self.lens = []
    if not is_generate_only:
      self.captions, self.cap2ftid = [], []
      for ftid, name in enumerate(self.names):
        self.captions.extend(self.ref_captions[name])
        self.cap2ftid.extend([ftid] * len(self.ref_captions[name]))
      self.cap2ftid = np.array(self.cap2ftid)
      self.num_caption = len(self.captions)
      self.print_fn('captions size %d' % self.num_caption)

      for i in range(len(self.captions)):
        self.lens.append(len(self.captions[i].split())+2)
    
    #according activitynet video name to get events seq. 
    self.video_names = []
    self.video_ft_list = []
    self.video_ents_len = []
    if is_use_eventsegment:
      self.full_video_dict = {}
      self.video_ft_len = {}
      for ftid, name in enumerate(self.names):
        dataset = self.name2dataset[ftid]
        if dataset == 'activitynet':
          name_split = name.split('_')
          keyname = '_'.join(name_split[:-2])
          start, end = int(name_split[-2]), int(name_split[-1])
          if keyname not in self.full_video_dict:  
            self.full_video_dict[keyname]=[[start, end]]
          else:
            self.full_video_dict[keyname].append([start, end])
      # sort segment time
      if task == 'eg' or task =='mem':
        self.event_list = []
        for dataset, name_file in name_files:
          if dataset != 'activitynet':
            continue
          current_data = np.load(name_file)
          event_val_e_dict = {}
          for ft_name in current_data:
            name_split = ft_name.split('_')
            keyname = '_'.join(name_split[:-2])
            start, end = int(name_split[-2]), int(name_split[-1])
            if keyname not in event_val_e_dict:
              event_val_e_dict[keyname] = [[start,end]]
            else:
              event_val_e_dict[keyname].append([start,end])
          self.event_list.append(event_val_e_dict)
        
        if len(self.event_list)==2:
          for key in self.event_list[0]:
            if key not in self.event_list[1]:
              self.event_list[1][key] = []
          for key in self.event_list[1]:
            if key not in self.event_list[0]:
              self.event_list[0][key] = []
          
      for key in self.full_video_dict:
        self.video_names.append(key)
        self.full_video_dict[key].sort(key=lambda x:[x[0],x[1]])
        if task == 'eg' or "mem" in task or task=='vem' or task =='mlm':
          ft = []
          for root in ft_roots['activitynet']:
            ft.append(np.load(os.path.join(root, '%s.mp4.npy'%key)))
          ft = np.concatenate(ft, axis=-1)
          self.video_ft_len[key] = len(ft)
          self.video_ft_list.append(len(ft))
          self.video_ents_len.append(len(self.full_video_dict[key]))
      
      if task == 'mem':
        self.video_names = []
        self.full_video_list = [] 
        for e_ent_dict in self.event_list:
          for key in e_ent_dict:
            if len(e_ent_dict[key]) > 0:
              self.video_names.append(key)
              e_ent_dict[key].sort(key=lambda x:[x[0],x[1]]) 
              self.full_video_list.append(e_ent_dict[key])
        self.print_fn('full_ent_list size %d' % len(self.full_video_list))


      self.num_video = len(self.video_names)
      self.print_fn('num_video size %d' % self.num_video)
    
    self.mem_reverse = mem_reverse
    if task == 'mem' or task =='vem' or task == 'eg' or task == 'sty':
      self.lens = []
      for i in range(len(self.video_names)):
        self.lens.append(16)
    self.has_sty = (sty_file !="")
    if (task == 'mem' or task == 'eg') and self.has_sty:
      self.sty_dict = json.load(open(sty_file))
    self.is_overlap_ft = is_overlap_ft
    self.stoi = json.load(open(word2int))
    self.itos = json.load(open(int2word))
    self.ft_roots = ft_roots
    self.max_words_in_sent = max_words_in_sent
    self.max_events_len = max_events_len
    self.max_ft_len = max_ft_len
    self.is_train = is_train
    self.task = task

    self.use_h5 = ('use_hdf5' in self.ft_roots) and self.ft_roots['use_hdf5']

    if self.use_h5:
      del self.ft_roots['use_hdf5']
      self.h5 = {}
      self.feat_dim = []
      for dataset, ft_list in self.ft_roots.items():
        self.h5[dataset] = [h5_file for h5_file in ft_list]
      self.feat_dim = [self.get_feat_dim(h5_i) for h5_i in self.h5[dataset]]
      

  def get_feat_dim(self, h5):
    if isinstance(h5, str):
      with h5py.File(h5, 'r') as f:
        return self.get_feat_dim(f)
    for key, value in h5.items():
      return value.shape[-1]


  def mask_events(self,x,task,max_len,is_ignore):
    ent_mask = []
    default_ft = np.zeros(self.max_ft_len)
    mask_ft = init_ft = end_ft = default_ft
    _real_len = min((x).shape[0],max_len-2)
    in_labels = []
    ignore_flag = []
    input = []
    _max_frame_include_num = np.zeros(self.max_ft_len)
    
    mask_whole_list = []
    _unique_start = np.unique(x[:,0])
    for e_idx in _unique_start:
      mask_whole_list.append(np.where(x[:,0]==e_idx)[0].tolist())

    is_overlap_ft = self.is_overlap_ft
    for i in range(0,_real_len):
      prob = random.random()
      ent_ft = np.zeros(self.max_ft_len)
      label_ent_ft = np.zeros(self.max_ft_len)
      label_ent_ft[x[i][0]:x[i][1]+1]=1
      _max_frame_include_num += label_ent_ft
      ent_ft[x[i][0]:x[i][1]+1]=_max_frame_include_num[x[i][0]:x[i][1]+1] if is_overlap_ft else label_ent_ft[x[i][0]:x[i][1]+1]
      if prob < 0.15 and ("mem" in task or task=='eg') and is_ignore:
        prob /= 0.15
        _ent_thres = 0.9 if task == 'mem' else 1.0
        if prob< _ent_thres:
          input.append(mask_ft)
        else:
          input.append(ent_ft)
        ent_mask.append(1)
        in_labels.append(label_ent_ft)
        ignore_flag.append(1)
      else:
        input.append(ent_ft)
        ent_mask.append(1)
        in_labels.append(label_ent_ft)
        ignore_flag.append(0)
  
    # head sequence need add event signal
    input.insert(0,init_ft)
    ent_mask.insert(0,1)
    in_labels.insert(0,init_ft)
    ignore_flag.insert(0,0)
    #add eos flag 
    prob = random.random()
    if len(in_labels) < max_len:
      if task == 'eg' and prob < 0.12:
        input.append(mask_ft)
        ent_mask.append(1)
        in_labels.append(end_ft)
        ignore_flag.append(1)
        
      else:
        input.append(end_ft)
        ent_mask.append(1)
        in_labels.append(end_ft)
        ignore_flag.append(0)

    else:
      if task == 'eg' and prob < 0.12:
        input[max_len-1] = mask_ft
        ent_mask[max_len-1] = 1
        in_labels[max_len-1] = end_ft
        ignore_flag.append(1)
      else:
        input[max_len-1] = end_ft
        ent_mask[max_len-1] = 1
        in_labels[max_len-1] = end_ft
        ignore_flag.append(0)

    right_offset = len(in_labels)
    for i in range(right_offset,max_len):
      input.append(end_ft)
      ent_mask.append(0)
      in_labels.append(end_ft)
      ignore_flag.append(0)
    return np.array(input,dtype=np.float32)[:max_len,:], np.array(ent_mask)[:max_len], np.array(in_labels,dtype=np.float32)[:max_len,:], np.array(ignore_flag,dtype=np.int32)[:max_len]
 

  def choice_random_events(self,idx,ft_len,ent_len,method = 0):
    if method==0:
      ft_lens = np.array(self.video_ft_list)
      request_idx = np.abs(ft_lens - ft_len).argsort()[:len(self.video_ft_list)//20]
      request_idx = np.setdiff1d(request_idx,idx)
      ent_lens = np.array(self.video_ents_len)
      request_idx2 = np.abs(ent_lens - ent_len).argsort()[:len(self.video_ents_len)//20]
      request_idx2 = np.array([x if x in request_idx else idx for x in request_idx2])
      request_idx2 = np.setdiff1d(request_idx2,idx)
      request_idx=np.concatenate((request_idx2, request_idx), axis=0)
      random_idx = random.choice(request_idx)
      name = self.video_names[random_idx]
    else:
      _total_idx = list(range(self.num_video))
      _total_idx.remove(idx)
      random_idx = random.choice(_total_idx)
      name = self.video_names[random_idx]
    event_times = np.array(self.full_video_dict[name],np.int32)
    indices = np.round(np.linspace(0, np.max(event_times), ft_len)).astype(np.int32)
    for i,e_ent in enumerate(event_times):
      e_ent[0] = np.abs(indices - e_ent[0]).argmin()
      e_ent[1] = np.abs(indices - e_ent[1]).argmin()
    return event_times

  def events_pad_or_trim_and_mask(self,events,max_len,is_ignore=True):
    if "mem" in self.task or self.task == 'vem' or self.task == 'eg':
      # mem tasks:masked events
      # ent_mask :1 represent pad.0 is the flag of masked 
      events_new, ent_mask, labels,ignore = self.mask_events(events,self.task, max_len,is_ignore)
    else:
      raise NotImplementedError('implement events data loader')
    return events_new,ent_mask,labels,ignore

  def temporal_pad_or_trim_feature(self, ft, max_len, event_times = None,transpose=False, average=False,video_cls = False):
    length, dim_ft = ft.shape
    new_ft_len = length
    # pad
    if length < max_len:
      ft_new = np.zeros((max_len, dim_ft), np.float32)
      ft_new[:length] = ft
      if video_cls:
        new_ft_len = new_ft_len+1
    # trim
    else:
      new_ft_len = max_len
      if average:
        indices = np.round(np.linspace(0, length, max_len+1)).astype(np.int32)
        ft_new = [np.mean(ft[indices[i]: indices[i+1]], axis=0) for i in range(max_len)]
        ft_new = np.array(ft_new, np.float32)
        if event_times is not None:
          for i,e_ent in enumerate(event_times):
            e_ent[0] = np.abs(indices - e_ent[0]).argmin()
            e_ent[1] = np.abs(indices - e_ent[1]).argmin()
      else:
        if video_cls:
          # ft[max_len-1] =[0]*len
          ft_new = np.zeros((max_len, dim_ft), np.float32)
          indices = np.round(np.linspace(0, length - 1, max_len-1)).astype(np.int32) 
          ft_new[:max_len-1] = ft[indices]
        else:
          indices = np.round(np.linspace(0, length - 1, max_len)).astype(np.int32)
          ft_new = ft[indices]
        # ft lengths sample,so the events must sample
        if event_times is not None:
          for i,e_ent in enumerate(event_times):
            e_ent[0] = np.abs(indices - e_ent[0]).argmin()
            e_ent[1] = np.abs(indices - e_ent[1]).argmin()

    if transpose:
      ft_new = ft_new.transpose()
    return np.array(ft_new,np.float32),event_times[:],new_ft_len

  def mask_video(self,ft,event_duration):
    length, dim_ft = ft.shape
    video_label = ft.copy()
    video_ft = ft.copy()
    ignore_flag =np.zeros(length)
    mask_ft = np.zeros(dim_ft, np.float32)
    for i in range(event_duration[0],min(event_duration[1]+1,self.max_ft_len)):
      prob = random.random()
      if prob < 0.15:
        video_ft[i] = mask_ft 
        ignore_flag[i] = 1
    return video_ft, video_label, np.array(ignore_flag,dtype=np.int32)

  def pad_sent(self, x):
    max_len = self.max_words_in_sent
    if 'mlm' in self.task or self.task == 'vc':  # masking language modeling or adapt to video captioning
      x, output_label = self.mask_sent(x[:max_len-1])
    else:
      output_label = [PAD] * (max_len-1)
    # padding and adding <sos> and <eos>
    prob = random.random()
    if self.task == 'vc' and prob < 0.12:   # mask <eos> for caption generation
      padded = [BOS] + x[:max_len-1] + [MASK] + [PAD] * max(0, max_len - len(x) - 2)
      output_label = [PAD] + output_label + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
      length = min(len(x)+2, max_len)
    elif self.task == 'ecm' or 'mlm' in self.task:
      padded = [CLS]+[BOS] + x[:max_len-1] + [EOS] + [PAD] * max(0, max_len - len(x) - 3)
      output_label = [PAD]*2 + output_label + [PAD] + [PAD] * max(0, max_len - len(x) - 3)
      length = min(len(x)+3, max_len)
    else:
      padded = [BOS] + x[:max_len-1] + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
      output_label = [PAD] + output_label + [PAD] + [PAD] * max(0, max_len - len(x) - 2)
      length = min(len(x)+2, max_len)
    
    # clip with max_len
    padded = padded[:max_len]
    output_label = output_label[:max_len]
    return np.array(padded), np.array(output_label), length

  def random_mask(self, x, i, prob):
    if prob < 0.8:
      x[i] = MASK
    elif prob < 0.9:
      x[i] = random.choice(list(range(len(self.stoi))))
    return x

  def mask_sent(self, x):
    output_label = []
    for i, token in enumerate(x):
      prob = random.random()
      if prob < 0.15:
        prob /= 0.15
        x = self.random_mask(x, i, prob)
        output_label.append(token)
      else:
        output_label.append(PAD)
    return x, output_label

  def cpt2int(self, cpt_list):
    int_cpt = [self.stoi.get(w, UNK) for w in cpt_list]
    return np.array(int_cpt)

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent.split()]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>', '<mask>')
    batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def int2eventlr(self,batch,names):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    ent_raw = []
    event_rst_batch = []
    max_ent_len_list = []
    for name in names:
      single_val_ent = []
      for val_ent in self.event_list:
        _e_ent = val_ent[name]
        single_val_ent.append(_e_ent)
      ent_raw.append(single_val_ent)
      ft_max = self.video_ft_len[name]
      max_ent_len_list.append(ft_max)
    for i in range (len(batch)):
      each_event = batch[i]
      _each_rst = []
      for e_segment in each_event:
        e_seg_np = np.array(e_segment)
        _idx_1_pred = np.where(e_seg_np==1)[0]
        if _idx_1_pred.shape[0] == 0:
          _idx_1_pred = np.array([0])
          break
        _each_rst.append([int(np.min(_idx_1_pred)),int(np.max(_idx_1_pred))])
      event_rst_batch.append(_each_rst)
    new_batch = []
    for ft_max,e_batch in zip(max_ent_len_list,event_rst_batch):
      if ft_max > self.max_ft_len:
        indices = np.round(np.linspace(0, ft_max - 1, self.max_ft_len)).astype(np.int32)
        for i,e_ent in enumerate(e_batch):
          e_ent[0] = int(indices[e_ent[0]])
          e_ent[1] = int(indices[e_ent[1]])
      new_e_batch = []
      for i,e_ent in enumerate(e_batch):
        e_ent[1] = e_ent[1] if e_ent[1] <= ft_max else ft_max
        if e_ent[0] <= e_ent[1] and e_ent[0] < ft_max:
          new_e_batch.append(e_ent)
      new_batch.append(new_e_batch)
    return new_batch,ent_raw,max_ent_len_list

  def __len__(self):
    if (self.is_train and self.task == 'vc')or self.task == 'mlm' or self.task == 'ecm':
      return self.num_caption
    elif self.task == 'mem' or self.task =='vem' or self.task == 'eg' or self.task=='sty':
      return self.num_video
    else:
      return self.num_ft
      
  def pad_feat(self, feats):
    max_len = max([len(l) for l in feats])
    for i in range(len(feats)):
      if len(feats[i]) < max_len:
        ft_new = np.zeros((max_len, feats[i].shape[-1]), np.float32)
        ft_new[:len(feats[i])] = feats[i]
        feats[i] = ft_new
    return feats

  def mem_getitem(self,idx):
    name = self.video_names[idx]
    outs = {}
    ft = []
    for root in self.ft_roots['activitynet']:
      ft.append(np.load(os.path.join(root, '%s.mp4.npy'%name)))
    ft = np.concatenate(ft, axis=-1)
    ft_len = min(self.max_ft_len, len(ft))
    self.video_ft_len[name] = len(ft)
    if self.task == 'mem':
      event_times = np.array(self.full_video_list[idx],np.int32) 
    else:
      event_times = np.array(self.full_video_dict[name],np.int32)
    ft,event_times,ft_len= self.temporal_pad_or_trim_feature(ft, self.max_ft_len,event_times=event_times,video_cls=False)
    if self.mem_reverse:
      new_events,ent_masks,out_label,ignore_label= self.events_pad_or_trim_and_mask(event_times[::-1],self.max_events_len)
    else:
      new_events,ent_masks,out_label,ignore_label= self.events_pad_or_trim_and_mask(event_times,self.max_events_len)
    if self.has_sty:
      video_cls = self.sty_dict[name]
      outs['video_cls'] = video_cls
    outs['ent_masks'] = ent_masks
    outs['ignore'] = ignore_label
    outs['ft_len'] = ft_len
    outs['cls'] = np.array([CLS])
    outs['video_ft'] = ft
    outs['names'] = name
    outs['event_times'] = new_events
    outs['output_label'] = out_label
    return outs
  

  def mlm_ent_pad_for_mem(self,event_single,event_real,video_name):
    _total_ent = self.full_video_dict[video_name] 
    final_idx = 0 
    for idx in range(len(_total_ent)):
      if event_single == _total_ent[idx]:
        final_idx = idx
        break
    rst_ent_ft = np.zeros((self.max_events_len,self.max_ft_len))
    rst_ent_mask = np.zeros(self.max_events_len)
    final_idx = final_idx+1 if final_idx+1 < self.max_events_len-1 else self.max_events_len-2
    rst_ent_mask[final_idx] = 1
    rst_ent_ft[final_idx][event_real[0]:event_real[1]+1]=1 
    return np.array(rst_ent_ft,np.float32),rst_ent_mask

  def __getitem__(self, idx):
    if not self.is_train and self.task == 'vc':
      name = self.names[idx]
      dataset = self.name2dataset[idx]
    elif self.task == 'mem' or self.task == 'eg':
      return self.mem_getitem(idx)
    else:
      name = self.names[self.cap2ftid[idx]]
      dataset = self.name2dataset[self.cap2ftid[idx]]

    outs = {}
    ft = []
    if self.use_h5:
      for i, _ in enumerate(self.h5[dataset]):
        if isinstance(self.h5[dataset][i], str):
          self.h5[dataset][i] = h5py.File(self.h5[dataset][i], 'r')
        if name not in self.h5[dataset][i]:
          print(f'Feature No.{i} not found. Dataset: {dataset}. Video name: {name}')
          print('Use zero tensor instead')
          ft.append(np.zeros((1, self.feat_dim[i]), dtype=np.float32))
        else:
          ft.append(np.array(self.h5[dataset][i][name], dtype=np.float32))
    else:
      if dataset == 'activitynet':
        name_split = name.split('_')
        keyname = '_'.join(name_split[:-2])
        start, end = int(name_split[-2]), int(name_split[-1])

      for root in self.ft_roots[dataset]:
        if dataset == 'activitynet':
          ft.append(np.load(os.path.join(root, '%s.mp4.npy'%keyname)))
        elif 'trecvid_19' in dataset or 'trecvid_20' in dataset:
          ft.append(np.load(os.path.join(root, '%s.npy'%name.split('_')[-1])))
        else:
          ft.append(np.load(os.path.join(root, '%s.npy'%name)))

    try:
      ft = np.concatenate(ft, axis=-1)
    except ValueError:
      print(f'Length of feature mismatch.\n Dataset: {dataset} Name: {name}')
      print(f'Align index to longest one...')
      max_len = max([ft[i].shape[0] for i in range(len(ft))])
      for i in range(len(ft)):
        align_index = np.round(np.linspace(0, ft[i].shape[0] - 1, max_len)).astype('int32')
        ft[i] = ft[i][align_index]
      ft = np.concatenate(ft, axis=-1)
    
    if dataset != 'activitynet':
      start = 0
      end = len(ft)

    ft_len = min(self.max_ft_len, len(ft))
    ft,event_times,_ = self.temporal_pad_or_trim_feature(ft, self.max_ft_len,event_times=np.array([[start, end]]))
    if dataset == 'activitynet':
      ent,ent_mask = self.mlm_ent_pad_for_mem([start,end],event_times[0].tolist(),keyname)
    else:
      ent = np.zeros((self.max_events_len,self.max_ft_len),np.float32)
      ent_mask = np.zeros(self.max_events_len)
      ent[1][event_times[0][0]:event_times[0][1]+1]=1
      ent_mask[1] = 1
    outs['event_times'] =  ent
    outs['ent_masks'] = ent_mask 
    outs['ft_len'] = ft_len
    new_ft,video_label,video_ignore = self.mask_video(ft,event_times[0].tolist())
    outs['video_label'] = video_label
    outs['video_ignore'] = video_ignore
    outs['video_ft'] = ft
    outs['video_ft_mvm'] = new_ft
    outs['names'] = name
    
    if self.use_cpt:
      outs['concept_ids'] = self.cpt2int(self.concepts[name])
    if not self.is_generate_only:
      caption_id, mask_label, caption_len = self.pad_sent(self.sent2int(self.captions[idx]))
      outs['caption_ids'] = caption_id
      outs['caption_lens'] = caption_len
      outs['output_label'] = mask_label
      outs['ref_sents'] = self.captions[idx]
    return outs


class TokenBucketSampler(Sampler):
  def __init__(self, lens, bucket_size, batch_size, droplast=False, size_multiple=8, rank=0, world_size=1):
    self._lens = lens
    self._max_tok = batch_size
    self._bucket_size = bucket_size
    self._droplast = droplast
    self._size_mul = size_multiple
    self.rank = rank
    self.world_size = world_size

  def _create_ids(self):
    return list(range(len(self._lens)))[self.rank::self.world_size]

  def _sort_fn(self, i):
    return self._lens[i]

  def __iter__(self):
    ids = self._create_ids()
    random.shuffle(ids)
    buckets = [sorted(ids[i:i+self._bucket_size], key=self._sort_fn, reverse=True)
                for i in range(0, len(ids), self._bucket_size)]
    batches = []
    for bucket in buckets:
      max_len = 0
      batch_indices = []
      for indices in partition_all(self._size_mul, bucket):
        max_len = max(max_len, max(self._lens[i] for i in indices))
        if (max_len * (len(batch_indices) + self._size_mul)
          > self._max_tok):
          if not batch_indices:
            raise ValueError("max_tokens too small / max_seq_len too long")
          assert len(batch_indices) % self._size_mul == 0
          batches.append(batch_indices)
          batch_indices = list(indices)
          max_len = max(self._lens[i] for i in indices)
        else:
          batch_indices.extend(indices)
      if not self._droplast and batch_indices:
        batches.append(batch_indices)
    random.shuffle(batches)
    return iter(batches)

  def __len__(self):
    raise ValueError("NOT supported. ")


class MetaLoader(object):
  """ wraps multiple data loaders """
  def __init__(self, loaders, accum_steps=1):
    assert isinstance(loaders, dict)
    self.name2loader = {}
    self.name2iter = {}
    self.sampling_pools = []
    for n, l in loaders.items():
      if isinstance(l, tuple):
        l, r = l
      elif isinstance(l, torch.utils.data.DataLoader):
        r = 1
      else:
        raise ValueError()
      self.name2loader[n] = l
      self.name2iter[n] = iter(l)
      self.sampling_pools.extend([n]*r)
    self.accum_steps = accum_steps
    self.step = 0

  def __iter__(self):
    """ this iterator will run indefinitely """
    task = self.sampling_pools[0]
    while True:
      if self.step % self.accum_steps == 0:
        task = random.choice(self.sampling_pools)
        self.step += 1
        iter_ = self.name2iter[task]
        try:
          batch = next(iter_)
        except StopIteration:
          iter_ = iter(self.name2loader[task])
          batch = next(iter_)
          self.name2iter[task] = iter_

      yield task, batch