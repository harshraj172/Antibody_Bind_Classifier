import torch
import torch.nn.functional as F
from GCL.losses import *
from utils import *
import utils

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau
    
    def compute(self, anchor, sample, pos_mask, neg_mask, topk, *args, **kwargs):
        sim = utils._similarity(anchor, sample) / self.tau
        # Loss 
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        # Accuracy
        acc = accuracy(sim, topk)
        return -loss.mean(), acc[0]
