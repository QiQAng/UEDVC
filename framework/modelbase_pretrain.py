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
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import framework.logbase
import pdb


class AdamW(Optimizer):
  """ Implements Adam algorithm with weight decay fix.
  Parameters:
    lr (float): learning rate. Default 1e-3.
    betas (tuple of 2 floats): Adams beta parameters (b1, b2).
        Default: (0.9, 0.999)
    eps (float): Adams epsilon. Default: 1e-6
    weight_decay (float): Weight decay. Default: 0.0
    correct_bias (bool): can be set to False to avoid correcting bias
        in Adam (e.g. like in Bert TF repository). Default True.
  """
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
               weight_decay=0.0, correct_bias=True):
    if lr < 0.0:
      raise ValueError(
        "Invalid learning rate: {} - should be >= 0.0".format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter: {} - "
                       "should be in [0.0, 1.0[".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter: {} - "
                       "should be in [0.0, 1.0[".format(betas[1]))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {} - "
                       "should be >= 0.0".format(eps))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                    correct_bias=correct_bias)
    super(AdamW, self).__init__(params, defaults)

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError(
            'Adam does not support sparse '
            'gradients, please consider SparseAdam instead')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
        denom = exp_avg_sq.sqrt().add_(group['eps'])

        step_size = group['lr']
        if group['correct_bias']:  # No bias correction for Bert
          bias_correction1 = 1.0 - beta1 ** state['step']
          bias_correction2 = 1.0 - beta2 ** state['step']
          step_size = (step_size * math.sqrt(bias_correction2)
                       / bias_correction1)

        p.data.addcdiv_(-step_size, exp_avg, denom)

        # Just adding the square of the weights to the loss function is
        # *not* the correct way of using L2 regularization/weight decay
        # with Adam, since that will interact with the m and v
        # parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't
        # interact with the m/v parameters. This is equivalent to
        # adding the square of the weights to the loss with plain
        # (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if group['weight_decay'] > 0.0:
          p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    return loss


class ModelBase(object):
  def __init__(self, config, _logger=None, gpu_id=0):
    '''initialize model 
    (support single GPU, otherwise need to be customized)
    '''
    self.device = torch.device("cuda:%d"%gpu_id if torch.cuda.is_available() else "cpu")
    self.config = config
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.submods = self.build_submods()
    for _, submod in self.submods.items():
      submod.to(self.device)
    self.criterion = self.build_loss()
    self.params, self.optimizer, self.lr_scheduler = self.build_optimizer()
    
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
    
  def validate(self, val_reader, TRG, refs):
    self.eval_start()
    # raise NotImplementedError('implement validate function: return metrics')

  def test(self, tst_reader, TRG, refs, tst_pred_file, tst_model_file=None):
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
        param_optimizer = list(submod.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        per_param_opts.append({
          'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
          'weight_decay': 0.01,
          'lr': self.config.base_lr * self.config.subcfgs[key].lr_mult
          })
        per_param_opts.append({
          'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
          'weight_decay': 0.0,
          'lr': self.config.base_lr * self.config.subcfgs[key].lr_mult
          })
        trn_params.extend([p for n, p in param_optimizer])
    if len(trn_params) > 0:
      #optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr)
      #optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr, betas=(0.9, 0.98), eps=1e-9)
      optimizer = AdamW(per_param_opts, lr=self.config.base_lr, betas=(0.9, 0.98), eps=1e-9)
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
    torch.save(state_dicts, ckpt_file)

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
    loss = self.forward_loss(batch_data, TRG, task=task, step=step)
    loss.backward()
    if step in [8000, 16000, 24000, 32000, 40000]:
      lr = self.optimizer.param_groups[0]['lr'] / 2
      self.optimizer.param_groups[0]['lr'] = lr
    #lr = 512 ** (-0.5) * min(step ** (-0.5), step * 8000**(-1.5))
    #self.optimizer.param_groups[0]['lr'] = lr
    self.optimizer.step()

    loss_value = loss.data.item()
    if self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
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

      if self.config.save_iter > 0 and step % self.config.save_iter == 0:
        self.save_checkpoint(os.path.join(model_dir, 'step.%d.th'%step))
      
      if (self.config.save_iter > 0 and step % self.config.save_iter == 0) \
        or (self.config.val_iter > 0 and step % self.config.val_iter == 0):
        metrics = self.validate(val_reader)
        with open(os.path.join(log_dir, 'val.step.%d.json'%step), 'w') as f:
          json.dump(metrics, f, indent=2)
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
      if epoch % 2 != -1:
        if self.config.save_per_epoch:
          self.save_checkpoint(os.path.join(model_dir, 'epoch.%d.th'%epoch))
        
        if self.config.val_per_epoch:
          metrics = self.validate(val_reader)
          with open(os.path.join(log_dir, 
            'val.epoch.%d.step.%d.json'%(epoch, step)), 'w') as f:
            json.dump(metrics, f, indent=2)
          self.pretty_print_metrics('epoch (%d/%d) val' % (epoch, self.config.num_epoch), metrics)
          # Write validation result into summary
          '''
          for metric_name, metric_score in metrics.items():
            self.tf_logger.scalar_summary('val_%s'%metric_name, metric_score, step)
          '''
          torch.cuda.empty_cache()
