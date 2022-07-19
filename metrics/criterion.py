import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import framework.configbase
import framework.ops
from cap_eval.cider.cider import Cider
from cap_eval.bleu.bleu import Bleu
from cap_eval.meteor.meteor import Meteor


class LabelSmoothingLoss(nn.Module):
  """
  With label smoothing,
  KL-divergence between q_{smoothed ground truth prob.}(w)
  and p_{prob. computed by model}(w) is minimized.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    assert 0.0 < label_smoothing <= 1.0
    self.padding_idx = ignore_index
    super(LabelSmoothingLoss, self).__init__()
    smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    one_hot = torch.full((tgt_vocab_size,), smoothing_value).cuda()
    one_hot[self.padding_idx] = 0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))
    self.confidence = 1.0 - label_smoothing

  def forward(self, output, target, norm):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    model_prob = self.one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
    loss = F.kl_div(output, model_prob, reduction='sum')
    return loss.div(float(norm))


class RewardLoss(nn.Module):
  def __init__(self, config=None):
    super(RewardLoss,self).__init__()
    self.config = config
    self.scorers = {}
    self.scorer_names = []
    if config is None:
      self.scorers['cider'] = Cider()
      self.scorers['meteor'] = Meteor()
      self.scorer_names = ['meteor']
      self.weights = np.array([5])
    else:
      self.weights = np.array(self.config['weights'])
      for scorer in self.config['scorers']:
        self.scorer_names.append(scorer)
        if scorer == 'cider':
          self.scorers['cider'] = Cider()
        elif scorer == 'bleu':
          self.scorers['bleu'] = Bleu(4)
        else:
          raise NotImplementedError

     

  def calc_reward(self, greedy_sents, sample_sents, ref_sents, len_size):
    batch_size = len(greedy_sents)
    rewards = np.zeros(shape=[batch_size,len_size], dtype=np.float32)
    for scorer_name, weight in zip(self.scorer_names, self.weights):
      scorer = self.scorers[scorer_name]
      if scorer_name == 'meteor':
        _, greedy_scores = scorer.compute_score(ref_sents, greedy_sents, vid_order=np.arange(batch_size))
        _, sample_scores = scorer.compute_score(ref_sents, sample_sents, vid_order=np.arange(batch_size))
      else:
        _, greedy_scores = scorer.compute_score(ref_sents, greedy_sents)
        _, sample_scores = scorer.compute_score(ref_sents, sample_sents)
      greedy_scores = np.array(greedy_scores)
      sample_scores = np.array(sample_scores)
      if scorer_name == 'bleu':
        greedy_scores = greedy_scores[-1]
        sample_scores = sample_scores[-1]
      rewards += np.expand_dims(weight * (sample_scores - greedy_scores),1)
    rewards = torch.FloatTensor(rewards).cuda().data
    return rewards

  def forward(self, sample_word_logprobs, sample_word_masks, greedy_sents, sample_sents, ref_sents):
    rewards = self.calc_reward(greedy_sents, sample_sents, ref_sents, sample_word_logprobs.size(1))
    logprobs = torch.sum(sample_word_logprobs * rewards * sample_word_masks)
    loss = - logprobs / torch.sum(sample_word_masks)
    return loss
    

class MultilabelCategoricalLoss(nn.Module):
  def __init__(self):
    super(MultilabelCategoricalLoss,self).__init__()

  def forward(self, y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class ContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0.2, max_violation=True, direction=2):
    '''Args:
      direction: 0 for negative sentence, 1 for negative image, 2 for both
    '''
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction

  def forward(self, scores):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
      the same row of im and s are positive pairs, different rows are negative pairs
    '''
    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    mask = torch.eye(batch_size).byte().to(scores.device)

    if self.direction == 0 or self.direction == 2:
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (self.margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill_(mask, 0)
      if self.max_violation:
        cost_s, _ = torch.max(cost_s, 1)
        cost_s = cost_s / batch_size
      else:
        cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 1 or self.direction == 2:
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (self.margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill_(mask, 0)
      if self.max_violation:
        cost_im, _ = torch.max(cost_im, 0)
        cost_im = cost_im / batch_size
      else:
        cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 0:
      return cost_s
    elif self.direction == 1:
      return cost_im
    else:
      return cost_s + cost_im