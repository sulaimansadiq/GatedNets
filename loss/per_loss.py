import torch
import torch.nn.functional as F
from torch import nn

from model.gated_conv import StraightThroughEstimator


class PerformanceLoss(nn.Module):
    def __init__(self, lam=0.0, gate_loss='l2'):
        super(PerformanceLoss, self).__init__()
        self.lam = lam
        self.gate_loss = gate_loss

    def forward(self, logits, targets, gates, gates_target, total_filters):

        ste = StraightThroughEstimator()
        #print("lam: ", lam)

        #compute cross entropy training loss
        ce_loss   = F.cross_entropy(logits, targets) #F.CrossEntropyLoss(logits, targets)

        layer_wise_gates = ste(torch.cat(gates[:], dim=1))
        gates = torch.sum(layer_wise_gates, dim=1)/total_filters

        if self.gate_loss == 'l2':
            gates_diff = torch.exp(self.lam*torch.pow(gates - gates_target, 2))
        elif self.gate_loss == 'l1':
            gates_diff = torch.exp(self.lam*torch.abs(gates - gates_target, 2))
        else:
            raise EnvironmentError

        # gates_diff = torch.exp(lam*torch.abs(gates - gates_target))
        gates_diff = torch.sum(gates_diff)/gates_diff.shape[0]  # average over batch size

        tot_loss = ce_loss*gates_diff

        return tot_loss, ce_loss, gates_diff

    def lam_update(self, num_epoch):      # might need to have a gate_loss_weight scheduler later
        self.lam = self.lam         # at the moment, only support static gate_loss_weight

    def get_curr_gate_loss_weight(self):
        return self.lam
