from __future__ import print_function
from __future__ import division

import os
import time
import json
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import framework.logbase
import pdb

#from apex import amp
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelBase(object):
  def __init__(self, config, _logger=None, gpu_id=0, rank=0, world_size=1):
    '''initialize model 
    (support single GPU, otherwise need to be customized)
    '''
    self.rank = rank
    self.device = torch.device("cuda:%d"%gpu_id if torch.cuda.is_available() else "cpu")
    self.config = config
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.submods = self.build_submods()
    for key, submod in self.submods.items():
      submod.to(self.device)
      if world_size > 1:
        self.submods[key] = DDP(submod, device_ids=[gpu_id], find_unused_parameters=True)
    self.criterion = self.build_loss()
    self.params, self.optimizer, self.lr_scheduler = self.build_optimizer()
    if hasattr(config, 'fp16') and config.fp16:
      self.scaler = GradScaler()
      # for model_key in self.submods.keys():
      #   self.submods[model_key], self.optimizer = amp.initialize(self.submods[model_key], self.optimizer, opt_level='O2')
    
    num_params, num_weights = 0, 0
    for key, submod in self.submods.items():
      for varname, varvalue in submod.state_dict().items():
        self.print_fn('%s: %s, shape=%s, num:%d' % (
          key, varname, str(varvalue.size()), np.prod(varvalue.size())))
        num_params += 1
        num_weights += np.prod(varvalue.size())
    self.print_fn('num params %d, num weights %d'%(num_params, num_weights))
    self.print_fn('trainable: num params %d, num weights %d'%(
      len(self.params), sum([np.prod(param.size()) for param in self.params])))

  def build_submods(self):
    raise NotImplementedError('implement build_submods function: return submods')

  def build_loss(self):
    raise NotImplementedError('implement build_loss function: return criterion')

  def forward_loss(self, batch_data, step=None, epoch=None):
    raise NotImplementedError('implement forward_loss function: return loss and additional outs')
    
  def validate(self, val_reader):
    self.eval_start()
    # raise NotImplementedError('implement validate function: return metrics')

  def test(self, tst_reader, tst_pred_file, tst_model_file=None):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()
    # raise NotImplementedError('implement test function')

  ########################## boilerpipe functions ########################
  def build_optimizer(self):
    trn_params = []
    per_param_opts = []
    for key, submod in self.submods.items():
      if self.config.subcfgs[key].freeze:
        for param in submod.parameters():
          param.requires_grad = False
      else:
        for param in submod.parameters():
          param.requires_grad = True
        params = []
        for name, param in submod.named_parameters():
          '''
          if 'cap_core' in name or 'img2state' in name or 'logit1' in name:
            param.requires_grad = False
          '''
          if param.requires_grad:
            params.append(param)
        per_param_opts.append({
          'params': params, 
          'lr': self.config.base_lr * self.config.subcfgs[key].lr_mult
          })
        trn_params.extend(params)
    if len(trn_params) > 0:
      if self.config.stage == 'finetune':
        optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr)
      else:
        optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr, betas=(0.9, 0.98), eps=1e-9)
      lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=self.config.decay_boundarys, gamma=self.config.decay_rate)
    else:
      optimizer, lr_scheduler = None, None
      print('no traiable parameters')
    return trn_params, optimizer, lr_scheduler

  def train_start(self):
    for key, submod in self.submods.items():
      submod.train()
    torch.set_grad_enabled(True)

  def eval_start(self):
    for key, submod in self.submods.items():
      submod.eval()
    torch.set_grad_enabled(False)

  def save_checkpoint(self, ckpt_file, submods=None):
    if submods is None:
      submods = self.submods
    state_dicts = {}
    for key, submod in submods.items():
      state_dicts[key] = {}
      for varname, varvalue in submod.state_dict().items():
        state_dicts[key][varname] = varvalue.cpu()
    torch.save(state_dicts, ckpt_file, _use_new_zipfile_serialization=False)

  def load_checkpoint(self, ckpt_file, submods=None):
    if submods is None:
      submods = self.submods
    state_dicts = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    
    num_resumed_vars = 0
    for key, state_dict in state_dicts.items():
      if key in submods:
        own_state_dict = submods[key].state_dict()
        new_state_dict = {}
        for varname, varvalue in state_dict.items():
          if varname in own_state_dict:
            new_state_dict[varname] = varvalue
            num_resumed_vars += 1
          elif 'module.'+varname in own_state_dict:
            new_state_dict['module.'+varname] = varvalue
            num_resumed_vars += 1
          elif varname[:7] == 'module.' and varname[7:] in own_state_dict:
            new_state_dict[varname[7:]] = varvalue
            num_resumed_vars += 1
        own_state_dict.update(new_state_dict)
        submods[key].load_state_dict(own_state_dict)
    self.print_fn('number of resumed variables: %d'%num_resumed_vars)

    
  def pretty_print_metrics(self, prefix, metrics):
    metric_str = []
    for measure, score in metrics.items():
      metric_str.append('%s %.4f'%(measure, score))
    metric_str = ' '.join(metric_str)
    self.print_fn('%s: %s' % (prefix, metric_str))

  def train_one_batch(self, batch_data, TRG, task, step):
    self.optimizer.zero_grad()
    #import pdb;pdb.set_trace()
    if hasattr(self.config, 'fp16') and self.config.fp16:
      with autocast():
        loss = self.forward_loss(batch_data, TRG, task=task, step=step)
      self.scaler.scale(loss).backward()
    else:
      loss = self.forward_loss(batch_data, TRG, task=task, step=step)
      loss.backward()

    # if hasattr(self.config, 'fp16') and self.config.fp16:
    #   with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #     scaled_loss.backward()
    # else:
    #   loss.backward()

    if hasattr(self.config, 'lr_scheduler'):
      if self.config.lr_scheduler == 'noam':
        lr = 512 ** (-0.5) * min(step ** (-0.5), step * 8000**(-1.5))
        self.optimizer.param_groups[0]['lr'] = lr
      elif self.config.lr_scheduler == 'noam_4000':
        lr = 512 ** (-0.5) * min(step ** (-0.5), step * 4000**(-1.5))
        self.optimizer.param_groups[0]['lr'] = lr
      elif self.config.lr_scheduler == 'linear':
        lr = max(1e-6, 1e-4 * min(step/4000, 1/(self.config.maximum_steps-4000) * (self.config.maximum_steps - step)))
        self.optimizer.param_groups[0]['lr'] = lr
      elif self.config.lr_scheduler == 'keep':
        # keep learning rate unchanged
        pass
    elif self.config.stage == 'finetune':
      if step in [8000, 16000, 24000, 32000, 40000]:
        lr = self.optimizer.param_groups[0]['lr'] / 2
        self.optimizer.param_groups[0]['lr'] = lr
    else:
      lr = 512 ** (-0.5) * min(step ** (-0.5), step * 8000**(-1.5))
      self.optimizer.param_groups[0]['lr'] = lr

    if hasattr(self.config, 'fp16') and self.config.fp16:
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      self.optimizer.step()

    loss_value = loss.data.item()
    if self.rank==0 and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      self.print_fn('\ttrn step %d %s: %.4f' % (step, 'loss', loss_value))
    # if self.config.summary_iter > 0 and step % self.config.summary_iter == 0:
    #   self.tf_logger.scalar_summary('trn_loss', loss_value, step)
    return {'loss': loss_value}

  def train_one_epoch(self, step, trn_reader, val_reader, model_dir, log_dir):
    self.train_start()
 
    avg_loss, n_batches = {}, {}
    for task, batch_data in trn_reader:
      loss = self.train_one_batch(batch_data, trn_reader.name2loader[task].dataset, task, step)
      for loss_key, loss_value in loss.items():
        avg_loss.setdefault(loss_key, 0)
        n_batches.setdefault(loss_key, 0)
        avg_loss[loss_key] += loss_value
        n_batches[loss_key] += 1
      step += 1
            
      if self.config.maximum_steps > 0 and step >= self.config.maximum_steps:
        exit()

      if self.rank == 0 and self.config.save_iter > 0 and step % self.config.save_iter == 0:
        self.save_checkpoint(os.path.join(model_dir, 'step.%d.th'%step))
      
      if self.rank == 0 and ((self.config.save_iter > 0 and step % self.config.save_iter == 0) \
        or (self.config.val_iter > 0 and step % self.config.val_iter == 0)):
        metrics, pred_sents = self.validate(val_reader)
        with open(os.path.join(log_dir, 'val.step.%d.json'%step), 'w') as f:
          json.dump(metrics, f, indent=2)
        os.makedirs(os.path.join(log_dir, '..', 'pred', 'val'), exist_ok=True)
        with open(os.path.join(log_dir, '..', 'pred', 'val',
          'val.step.%d.json'%step), 'w') as f:
          json.dump({"rst":pred_sents}, f, indent=2)
        self.pretty_print_metrics('\tval step %d'%step, metrics)
        # Write validation result into summary
        '''
        for metric_name, metric_score in metrics.items():
          self.tf_logger.scalar_summary('val_%s'%metric_name, metric_score, step)
        '''
        self.train_start()


    for loss_key, loss_value in avg_loss.items():
      avg_loss[loss_key] = loss_value / n_batches[loss_key]
    return avg_loss, step

  def epoch_postprocess(self, epoch):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

  def train(self, trn_reader, val_reader, model_dir, log_dir, resume_file=None):
    assert self.optimizer is not None
   
    if resume_file is not None:
      self.load_checkpoint(resume_file)
    # tf logger
    #self.tf_logger = framework.logbase.TensorboardLogger(log_dir)
   
    # first validate
    #metrics = self.validate(val_reader, TRG, refs)
    #self.pretty_print_metrics('init val', metrics)
    
    # training
    step = 1
    for epoch in range(self.config.num_epoch):
      avg_loss, step = self.train_one_epoch(
        step, trn_reader, val_reader, model_dir, log_dir)
      self.pretty_print_metrics('epoch (%d/%d) trn'%(epoch, self.config.num_epoch), avg_loss)
      self.epoch_postprocess(epoch)
      if hasattr(self.config, 'lr_scheduler'):
        if self.config.lr_scheduler == 'epoch_down':
          if epoch % 19 == 0:
            lr = self.optimizer.param_groups[0]['lr'] / 10
            self.optimizer.param_groups[0]['lr'] = lr
            print(f'Epoch {epoch} lr down to {lr}')
      if self.rank == 0 and epoch % 2 != -1:
        if self.config.save_per_epoch:
          self.save_checkpoint(os.path.join(model_dir, 'epoch.%d.th'%epoch))
        
        if self.config.val_per_epoch:
          metrics, pred_sents = self.validate(val_reader)
          with open(os.path.join(log_dir, 
            'val.epoch.%d.step.%d.json'%(epoch, step)), 'w') as f:
            json.dump(metrics, f, indent=2)
          os.makedirs(os.path.join(log_dir, '..', 'pred', 'val'), exist_ok=True)
          with open(os.path.join(log_dir, '..', 'pred', 'val',
            'val.epoch.%d.step.%d.json'%(epoch, step)), 'w') as f:
            json.dump(pred_sents, f, indent=2)
          self.pretty_print_metrics('epoch (%d/%d) val' % (epoch, self.config.num_epoch), metrics)
          # Write validation result into summary
          '''
          for metric_name, metric_score in metrics.items():
            self.tf_logger.scalar_summary('val_%s'%metric_name, metric_score, step)
          '''
          torch.cuda.empty_cache()
