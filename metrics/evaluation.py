import numpy as np
import torch
from cap_eval.bleu.bleu import Bleu
from cap_eval.meteor.meteor import Meteor
from cap_eval.cider.cider import Cider

meteor_scorer = Meteor()
cider_scorer = Cider()
bleu_scorer = Bleu(4)

def iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou

def compute_mem_lr(preds,targets):
  assert preds.size()==targets.size()
  #import pdb
  #pdb.set_trace()
  preds = preds.data.cpu().numpy()
  targets = targets.data.cpu().numpy()
  preds = (preds>0.5)
  mask_true = 0 
  mask_num = 0
  pred_ents = []
  target_ents = []
  for e_pred,e_target in zip(preds,targets):
    _idx_1_pred = np.where(e_pred==1)[0]
    _idx_1_gt = np.where(e_target==1)[0]
    if _idx_1_pred.shape[0] == 0:
      _idx_1_pred = np.array([0])
    if _idx_1_gt.shape[0] == 0:
      _idx_1_gt = np.array([0])
    pred_ents.append([np.min(_idx_1_pred),np.max(_idx_1_pred)+1])
    target_ents.append([np.min(_idx_1_gt),np.max(_idx_1_gt)+1])
  _iou_sum = 0
  iou_threshold = np.array([0.3,0.5,0.7,0.9])
  _iou_true = np.zeros(4,np.float)
  for e_pred,e_target in zip(pred_ents,target_ents):
    _ratio = iou(e_pred,e_target)
    _iou_sum += _ratio
    _true_bool = (_ratio >= iou_threshold)
    _iou_true  = _iou_true + _true_bool
  mask_true = (np.array(pred_ents).reshape(-1)==np.array(target_ents).reshape(-1)).sum()
  mask_num = len(pred_ents)*2
  return _iou_sum,_iou_true,mask_true,mask_num

def compute_tiou(preds,targets):
  """
  preds:1-dims
  targets:1-dims
  """
  assert preds.size()==targets.size()
  preds = preds.view(-1,2)
  targets = targets.view(-1,2)
  _iou_sum = 0
  iou_threshold = np.array([0.3,0.5,0.7,0.9])
  _iou_true = np.zeros(4,np.float)
  for e_pred,e_target in zip(preds,targets):
    _ratio = iou(e_pred,e_target).item()
    _iou_sum += _ratio
    _true_bool = (_ratio >= iou_threshold)
    _iou_true  = _iou_true + _true_bool
  return _iou_sum, preds.size(0), _iou_true

def evaluate_detection(preds,targets,tiou):
  recall = [0] * len(preds)
  precision = [0] * len(preds)
  for vid_i,(e_pred,e_target) in enumerate(zip(preds, targets)):
    best_recall = 0
    best_precision = 0
    for gt_ent in e_target:
      ref_set_covered = set([])
      pred_set_covered = set([])
      pred_i = 0
      for pred_i, pred in enumerate(e_pred):
        for ref_i, ref_timestamp in enumerate(gt_ent):
          if iou(pred, ref_timestamp) > tiou:
            ref_set_covered.add(ref_i)
            pred_set_covered.add(pred_i)
      new_precision = float(len(pred_set_covered)) / (pred_i + 1)
      best_precision = max(best_precision, new_precision)
      new_recall = float(len(ref_set_covered)) / len(gt_ent) if len(gt_ent) > 0 else 0
      best_recall = max(best_recall, new_recall)
    recall[vid_i] = best_recall
    precision[vid_i] = best_precision
  return sum(precision) / len(precision), sum(recall) / len(recall)

def compute_tiou_for_eg(preds,targets):
  iou_threshold = [0.3,0.5,0.7,0.9]
  _recall = np.zeros(4,np.float)
  _percision = np.zeros(4,np.float)
  for i,tiou in enumerate(iou_threshold):
    _percision[i],_recall[i] = evaluate_detection(preds,targets,tiou)
  
  return _recall,_percision

def bleu_eval(refs, cands):
  print ("calculating bleu_4 score...")
  bleu, _ = bleu_scorer.compute_score(refs, cands)
  return bleu

def cider_eval(refs, cands):
  print ("calculating cider score...")
  cider, _ = cider_scorer.compute_score(refs, cands)
  return cider

def meteor_eval(refs, cands):
  print ("calculating meteor score...")
  meteor, _ = meteor_scorer.compute_score(refs, cands)
  return meteor

def compute(preds, refs, names):
  refcaps = {}
  candcaps = {}
  for i in range(len(preds)):
    candcaps[str(i)] = [preds[i]]
    refcaps[str(i)] = refs[names[i]]
  bleu = bleu_eval(refcaps, candcaps)
  cider = cider_eval(refcaps, candcaps)
  meteor = meteor_eval(refcaps, candcaps)
  scores = {'meteor':meteor,'cider':cider,'bleu_4':bleu[3],'bleu_3':bleu[2],'bleu_2':bleu[1],'bleu_1':bleu[0]}
  return scores

def eval_q2m(scores, q2m_gts, return_ranks=False, topk=1):
  '''
  Image -> Text / Text -> Image
  Args:
    scores: (n_query, n_memory) matrix of similarity scores
    q2m_gts: list, each item is the positive memory ids of the query id
  Returns:
    scores: (recall@1, 5, 10, median rank, mean rank)
    gt_ranks: the best ranking of ground-truth memories
    pred_topks: the predicted topk memory ids
  '''
  n_q, n_m = scores.shape
  gt_ranks = np.zeros((n_q, ), np.int32)
  pred_topks = np.zeros((n_q, topk), np.int32)

  for i in range(n_q):
    s = scores[i]
    sorted_idxs = np.argsort(-s)

    rank = n_m
    for k in q2m_gts[i]:
      tmp = np.where(sorted_idxs == k)[0][0]
      if tmp < rank:
        rank = tmp
      gt_ranks[i] = rank
      pred_topks[i] = sorted_idxs[:topk]

  # compute metrics
  r1 = 100 * len(np.where(gt_ranks < 1)[0]) / n_q
  r5 = 100 * len(np.where(gt_ranks < 5)[0]) / n_q
  r10 = 100 * len(np.where(gt_ranks < 10)[0]) / n_q
  medr = np.median(gt_ranks) + 1
  meanr = gt_ranks.mean() + 1
  if return_ranks:
    return (r1, r5, r10, medr, meanr), (gt_ranks, pred_topks)
  else:
    return (r1, r5, r10, medr, meanr)