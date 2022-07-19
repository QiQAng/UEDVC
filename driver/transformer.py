from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.abspath('..'))
import argparse
import json
import time
import random
import numpy as np

import torch
import torch.utils.data as data

import framework.run_utils
import framework.logbase
import models.transformer
from models.transformer import DECODER
import readers.data as dataset

import torch.distributed as dist
import torch.multiprocessing as mp

def getPort():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000,20000)
    if tt not in procarr:
        return tt
    else:
        return getPort()


def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def main(gpu, opts, path_cfg, model_cfg):

  torch.cuda.set_device(gpu)
  rank = opts.nr * opts.gpus + gpu
  if opts.world_size > 1:
    dist.init_process_group(
      backend='nccl',
      init_method='env://',
      world_size=opts.world_size,
      rank=rank
    )
  
  if rank == 0 and path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None

  _model = models.transformer.TransModel(model_cfg, _logger=_logger, gpu_id=gpu, rank=rank, world_size=opts.world_size)


  if opts.is_train:
    if rank == 0:
      model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
      path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
      json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_reader, val_reader = {}, {}
    for task in path_cfg.pretrain_task:
      if hasattr(path_cfg, 'concept_file'):
        trn_data = dataset.PretrainVCDataset(path_cfg.name_file['trn'], path_cfg.anno_file, path_cfg.concept_file, path_cfg.ft_root['trn'], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=True, task=task, _logger=_logger, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)
      else:
        trn_data = dataset.PretrainVCDataset(path_cfg.name_file['trn'], path_cfg.anno_file, None, path_cfg.ft_root['trn'], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=True, task=task, _logger=_logger, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)
      batch_size = 5000 if not hasattr(model_cfg, 'trn_max_token') else model_cfg.trn_max_token
      sampler = dataset.TokenBucketSampler(trn_data.lens, bucket_size=8192, batch_size=batch_size, size_multiple=8, rank=rank, world_size=opts.world_size)
      r = 1
      if task == 'mlm':
        r = model_cfg.subcfgs[DECODER].mlm_batch_loop_num
      elif task == 'mem':
        r = model_cfg.subcfgs[DECODER].mem_batch_loop_num
      trn_reader[task] = (data.DataLoader(trn_data, batch_sampler=sampler, num_workers=8), r)
    meta_loader = dataset.MetaLoader(trn_reader)

    for task in path_cfg.eval_task:
      if hasattr(path_cfg, 'concept_file'):
        val_data = dataset.PretrainVCDataset(path_cfg.name_file['val'], path_cfg.anno_file, path_cfg.concept_file, path_cfg.ft_root['val'], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=False, task=task, _logger=_logger, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)
      else:
        val_data = dataset.PretrainVCDataset(path_cfg.name_file['val'], path_cfg.anno_file, None, path_cfg.ft_root['val'], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=False, task=task, _logger=_logger, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)      
      val_reader[task] = data.DataLoader(val_data, batch_size=model_cfg.tst_batch_size, shuffle=False, num_workers=4)

    _model.train(meta_loader, val_reader, path_cfg.model_dir, path_cfg.log_dir, resume_file=opts.resume_file)

  else:
    tst_reader = {}
    is_generate_only = (opts.eval_set == 'tst')
    for task in path_cfg.eval_task:
      if hasattr(path_cfg, 'concept_file'):
        tst_data = dataset.PretrainVCDataset(path_cfg.name_file[opts.eval_set], path_cfg.anno_file, path_cfg.concept_file, path_cfg.ft_root[opts.eval_set], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=False, task=task, _logger=_logger, is_generate_only=is_generate_only, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)
      else:
        tst_data = dataset.PretrainVCDataset(path_cfg.name_file[opts.eval_set], path_cfg.anno_file, None, path_cfg.ft_root[opts.eval_set], 
          path_cfg.word2int_file, path_cfg.int2word_file, model_cfg.subcfgs[DECODER].max_words_in_sent, model_cfg.subcfgs[DECODER].max_ft_len, is_overlap_ft = model_cfg.subcfgs[DECODER].is_overlap_ft, sty_file=path_cfg.sty_file, mem_reverse=model_cfg.subcfgs[DECODER].mem_reverse, is_train=False, task=task, _logger=_logger, is_generate_only=is_generate_only, max_events_len = model_cfg.subcfgs[DECODER].max_events_len)
      tst_reader[task] = data.DataLoader(tst_data, batch_size=model_cfg.tst_batch_size, shuffle=False, num_workers=4)

    model_str_scores = []
    is_first_eval = True
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(path_cfg.log_dir, path_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      set_pred_dir = os.path.join(path_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.json')

      scores = _model.test(tst_reader, tst_pred_file, tst_model_file=model_file, do_eval=not is_generate_only)
      if is_first_eval:
        score_names = scores.keys()
        model_str_scores.append(','.join(score_names))
        is_first_eval = False
        print(model_str_scores[-1])
      str_scores = [measure_name, os.path.basename(model_file)]
      for score_name in score_names:
        str_scores.append('%.4f'%(scores[score_name]))
      str_scores = ','.join(str_scores)
      print(str_scores)
      model_str_scores.append(str_scores)

    score_log_file = os.path.join(path_cfg.pred_dir, opts.eval_set, 'scores.csv')
    with open(score_log_file, 'w') as f:
      for str_scores in model_str_scores:
        print(str_scores, file=f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', default=False, action='store_true')
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--eval_set', default='val')
  parser.add_argument('--generate_only', default=False, action='store_true')

  parser.add_argument('-n', '--nodes', default=1,
                      type=int, metavar='N')
  parser.add_argument('-g', '--gpus', default=1, type=int,
                      help='number of gpus per node')
  parser.add_argument('-nr', '--nr', default=0, type=int,
                      help='ranking within the nodes')

  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = str(getPort())

  opts = parser.parse_args()
  opts.world_size = opts.gpus * opts.nodes

  path_cfg = framework.run_utils.gen_common_pathcfg(
    opts.path_cfg_file, is_train=opts.is_train)
 
  model_cfg = models.transformer.TransModelConfig()
  model_cfg.load(opts.model_cfg_file)
  set_seeds(12345)
  # main(opts, path_cfg, model_cfg)
  if opts.world_size > 1:
    mp.spawn(main, nprocs=opts.gpus, args=(opts, path_cfg, model_cfg))
  else:
    main(0, opts, path_cfg, model_cfg)
