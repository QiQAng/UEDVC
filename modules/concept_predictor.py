import torch
import torch.nn as nn
from modules.common import *
import framework.configbase

class CPTMConfig(framework.configbase.ModuleConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.d_out = 1406
        self.dropout = 0.1

class ConceptPredictor(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        # self.ff = FeedForward(d_model, d_ff=d_model, dropout=0.1)
        # self.dropout = nn.Dropout(p=dropout)
        self.logit_layer = nn.Linear(d_model, d_out, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden):
        logits = self.logit_layer(hidden)
        probs = self.sigmoid(logits)
        return probs

