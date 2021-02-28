import torch
import torch.nn.functional as F
from torch import nn


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class GatingNW(nn.Module):

  def __init__(self, in_chs=None, out_gates=None):
    super(GatingNW, self).__init__()

    self.fc1 = nn.Linear(in_chs, out_gates)
    # add batchnorm here
    self.bn1 = nn.BatchNorm1d(out_gates)
    self.activ1 = nn.ReLU()
    self.fc2 = nn.Linear(out_gates, out_gates)
    # add batchnorm here
    self.bn2 = nn.BatchNorm1d(out_gates)
    self.ste = StraightThroughEstimator()

  def forward(self, x, bin=True):

    # print('Here1')
    out = torch.mean(x, dim=(2, 3))     # spatial average of channels to give 1x1

    out = out.view(out.shape[0], -1)      # flatten to in_chs dim vector
    out = self.fc1(out)                    # apply fc
    out = self.bn1(out)
    out = self.activ1(out)

    out = self.fc2(out)
    # out = self.bn2(out)
    # print('out.shape GNW: ', out.shape)
    if self.training:
      out = out + torch.normal(0, 1, out.shape).to(out.device)
      out_ss = torch.maximum(torch.zeros(out.shape, device=out.device), torch.minimum(torch.ones(out.shape, device=out.device), 1.2*torch.sigmoid(out) - 0.1))
      # out_ss = torch.sigmoid(out)
      if bin:
        # step function is non-linearity with STE
        out_bin = self.ste(out)
        # remove contribution of SS in output of network
        # and remove contribution of binarisation in gradients, keep SS contribution
        out = out_ss + out_bin.detach() - out_ss.detach()
      else:
        out = out_ss
    else:
        out = self.ste(out)

    # out_ss = torch.sigmoid(out)
    # out_bin = self.ste(out)
    # out = out_ss + out_bin.detach() - out_ss.detach()

    # print('out.shape GNW: ', out.shape)
    return out


class GatedConv2d(nn.Module):

    def __init__(self, in_chs=None, out_chs=None, ker_sz=None, pad=None, man_gates_=None):
        super(GatedConv2d, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(out_chs):  # every filter applied separately, so that the output channels can be gated
            op = nn.Conv2d(in_chs, 1, ker_sz, padding=pad)
            self.convs.append(op)
        self.bn1 = nn.BatchNorm2d(out_chs)

        self.num_gates = out_chs  # one gate for every filter
        self.gating_nw = GatingNW(in_chs=in_chs, out_gates=out_chs)

        # if gates == None:
        #     gates = torch.ones(self.num_gates)
        #     self.manual_gates = False
        # else:
        #     print('Manual Gating Mode')
        #     gates = torch.ones(self.num_gates)
        #     #      gates = torch.tensor(gates)
        #     self.manual_gates = True

        self.man_gates, self.man_on_gates = man_gates_
        self.gates = torch.nn.Parameter(torch.zeros(self.num_gates), requires_grad=False)
        self.gates[:self.man_on_gates] = 1.0
            # self.gates = torch.nn.Parameter(torch.ones(self.num_gates), requires_grad=False) # untrainable parameters for debugging

    # end of __init__()

    def forward(self, x, cond=True):  # , gates=None):
        gates = self.gating_nw(x, bin=cond)

        if self.man_gates:  # this doesnt need to be inside loop
            gates = self.gates.repeat(x.shape[0], 1)

        out_channels = []
        for i in range(len(self.convs)):  # apply all the convolutions separately
            out = self.convs[i](x)  # check whether this works correctly with sandbox example

            # if self.man_gates:                              # this doesnt need to be inside loop
            #     gates = self.gates.repeat(out.shape[0], 1)
            # g = gates[:, i]

            # print(g.shape)
            # out = out * g[:, None, None, None]  # apply gating mask
            out_channels.append(out)  # in eval append zeros

        out = torch.cat(out_channels[:], dim=1)
        out = self.bn1(out)                         # this is being applied after gating dumbass
        out = out*gates[:, :, None, None]
        return out, gates
    # end of forward()


