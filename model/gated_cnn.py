import torch
import torch.nn.functional as F
from torch import nn
from model.gated_conv import GatedConv2d


class GatedCNN(nn.Module):

    def __init__(self, debug_gates=None):
        super(GatedCNN, self).__init__()

        self.num_classes = 10
        self.bottle_sz = 10
        self.d_gates = debug_gates

        self.gconv1 = GatedConv2d(in_chs=4, out_chs=3, ker_sz=(3, 3), pad=0,
                                  gates=self.d_gates)  # perform image processing conv only on image data
        self.gconv2 = GatedConv2d(in_chs=3, out_chs=10, ker_sz=(3, 3), pad=0, gates=self.d_gates)
        # self.gconv3       = GatedConv2d(in_chs=3, out_chs=3, ker_sz=(3,3), pad=0, gates=self.d_gates)
        # self.gconv4       = GatedConv2d(in_chs=3, out_chs=10, ker_sz=(3,3), pad=0, gates=self.d_gates)

        self.fc1 = nn.Linear(2560, self.bottle_sz)
        self.fc2 = nn.Linear(self.bottle_sz, self.num_classes)

    def forward(self, x):
        # perform processing on image only
        # image, constraint = torch.split(x, [3, 1], dim=1)

        cond = torch.rand(1) < 0.5
        gates = []

        # print(x.shape)
        cnn_out, gate_nw_out = self.gconv1(x, cond)
        cnn_out = F.relu(cnn_out)
        gates.append(gate_nw_out)
        # print(cnn_out.shape)
        cnn_out, gate_nw_out = self.gconv2(cnn_out, cond)
        cnn_out = F.relu(cnn_out)
        gates.append(gate_nw_out)
        # # # print(cnn_out.shape)
        # cnn_out, gate_nw_out = self.gconv3(cnn_out, cond)
        # cnn_out = F.relu(cnn_out)
        # gates.append(gate_nw_out)
        # # # print(cnn_out.shape)
        # cnn_out, gate_nw_out = self.gconv4(cnn_out, cond)
        # cnn_out = F.relu(cnn_out)
        # gates.append(gate_nw_out)
        # print(cnn_out.shape)

        # cnn_out = torch.mean(cnn_out, dim=(2, 3))
        cnn_out = cnn_out.view(cnn_out.shape[0], -1)      # flatten to in_chs dim vector
        cnn_out = self.fc1(cnn_out)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.fc2(cnn_out)

        return cnn_out, gates, cond