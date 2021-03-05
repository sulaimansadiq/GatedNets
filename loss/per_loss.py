import torch
import torch.nn.functional as F
from torch import nn

from model.gated_conv import StraightThroughEstimator


class PerformanceLoss(nn.Module):
    def __init__(self, lam=0.0):

        super(PerformanceLoss, self).__init__()
        self.lam = lam

    def forward(self, logits, targets, gates, gates_target, total_filters):

        ste = StraightThroughEstimator()
        #print("lam: ", lam)

        #compute cross entropy training loss
        ce_loss   = F.cross_entropy(logits, targets) #F.CrossEntropyLoss(logits, targets)

        layer_wise_gates = ste(torch.cat(gates[:], dim=1))
        gates = torch.sum(layer_wise_gates, dim=1)/total_filters

        gates_diff = torch.exp(self.lam*torch.pow(gates - gates_target, 2))

        # gates_diff = torch.exp(lam*torch.abs(gates - gates_target))
        gates_diff = torch.sum(gates_diff)/gates_diff.shape[0]  # average over batch size

        tot_loss = ce_loss*gates_diff

        return tot_loss, ce_loss, gates_diff

    def lam_update(self, num_epoch):      # might need to have a gate_loss_weight scheduler later
        self.lam = self.lam         # at the moment, only support static gate_loss_weight

    def get_curr_gate_loss_weight(self):
        return self.lam


class PerformanceLoss_v2(nn.Module):
    def __init__(self, alpha=0.4, beta=1.3):

        super(PerformanceLoss_v2, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, gates, gates_target, total_filters):

        ste = StraightThroughEstimator()
        #print("lam: ", lam)

        #compute cross entropy training loss
        ce_loss   = F.cross_entropy(logits, targets)

        layer_wise_gates = ste(torch.cat(gates[:], dim=1))          # apply step function toall gates
        gates = torch.sum(layer_wise_gates, dim=1)/total_filters    # get total num of ON gates

        gates_diff = torch.exp(self.alpha * torch.pow(torch.abs(gates - gates_target), self.beta))     # MSE with gates target

        # gates_diff = torch.exp(lam*torch.abs(gates - gates_target))
        gates_diff = torch.sum(gates_diff)/gates_diff.shape[0]  # average over batch size

        tot_loss = ce_loss*gates_diff

        return tot_loss, ce_loss, gates_diff

