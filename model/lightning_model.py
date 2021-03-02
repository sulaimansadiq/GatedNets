import os

ENV_PATHS = ["H:\\mycode\\my_envs\\py3.6_torch1.7.1_",
             "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\mingw-w64\\bin",
             "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\usr\\bin",
             "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\bin",
             "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Scripts",
             "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\bin"]

for path in ENV_PATHS:
    if path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + path

import pytorch_lightning as pl
import torch
from torch import optim
from model.gated_cnn import GatedCNN
from model.gated_conv import GatedConv2d
from torch import nn
from loss.per_loss import PerformanceLoss


class LightningGatedCNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = GatedCNN(self.hparams)
        self.layer_filters = []
        for m1 in self.model.modules():
            f = 0
            if isinstance(m1, GatedConv2d):
                for m2 in m1.modules():
                    if isinstance(m2, nn.Conv2d):
                        f = f + 1
                self.layer_filters.append(f)
        self.total_filters = float(sum(self.layer_filters))   # make this dynamic
        self.norm_consts = [float(g)/self.total_filters for g in self.hparams['constraints']]
        self.num_consts = len(self.hparams['constraints'])
        self.accuracy = pl.metrics.Accuracy()

        if self.hparams['logging']:

            for i in range(len(self.model.gconv1.convs)):
                self.model.gconv1.convs[i].register_forward_hook(self.get_activation('fwd.model.gconv1.convs'+str(i)))
            self.model.gconv1.bn1.register_forward_hook(self.get_activation('fwd.model.gconv1.bn1'))
            self.model.activ1.register_forward_hook(self.get_activation('fwd.model.activ1'))

            for i in range(len(self.model.gconv2.convs)):
                self.model.gconv2.convs[i].register_forward_hook(self.get_activation('fwd.model.gconv2.convs'+str(i)))
            self.model.gconv2.bn1.register_forward_hook(self.get_activation('fwd.model.gconv2.bn1'))
            self.model.activ2.register_forward_hook(self.get_activation('fwd.model.activ2'))

            for i in range(len(self.model.gconv3.convs)):
                self.model.gconv3.convs[i].register_forward_hook(self.get_activation('fwd.model.gconv3.convs'+str(i)))
            self.model.gconv3.bn1.register_forward_hook(self.get_activation('fwd.model.gconv3.bn1'))
            self.model.activ3.register_forward_hook(self.get_activation('fwd.model.activ3'))

            self.model.gconv1.gating_nw.fc1.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw.fc1'))
            self.model.gconv1.gating_nw.activ1.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw.activ1'))
            # self.model.gconv1.gating_nw.bn1.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw.bn1'))
            self.model.gconv1.gating_nw.fc2.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw.fc2'))
            self.model.gconv1.gating_nw.bn2.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw.bn2'))
            self.model.gconv1.gating_nw.register_forward_hook(self.get_activation('fwd.model.gconv1.gating_nw'))

            self.model.gconv2.gating_nw.fc1.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw.fc1'))
            # self.model.gconv2.gating_nw.bn1.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw.bn1'))
            self.model.gconv2.gating_nw.activ1.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw.activ1'))
            self.model.gconv2.gating_nw.fc2.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw.fc2'))
            self.model.gconv2.gating_nw.bn2.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw.bn2'))
            self.model.gconv2.gating_nw.register_forward_hook(self.get_activation('fwd.model.gconv2.gating_nw'))

            self.model.gconv3.gating_nw.fc1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.fc1'))
            # self.model.gconv3.gating_nw.bn1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.bn1'))
            self.model.gconv3.gating_nw.activ1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.activ1'))
            self.model.gconv3.gating_nw.fc2.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.fc2'))
            self.model.gconv3.gating_nw.bn2.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.bn2'))
            self.model.gconv3.gating_nw.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw'))

    def forward(self, x):
        out = self.model(x)

    def configure_optimizers(self):
        # optimiser = optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])
        optimiser = optim.SGD(self.parameters(), self.hparams['learning_rate'], self.hparams['momentum'], self.hparams['weight_decay'])
        return optimiser

    def training_step(self, batch, batch_idx):
        input, target = batch

        # augmenting dataset here, very hacky, fix later do this in dataloader
        aug_input = input.repeat(self.num_consts, 1, 1, 1)
        cons = torch.repeat_interleave(torch.tensor(self.norm_consts, device=aug_input.device), input.shape[0]).repeat(input.shape[2], input.shape[3], 1, 1).T
        aug_input = torch.cat((aug_input, cons), 1)
        cons_target = torch.repeat_interleave(torch.tensor(self.norm_consts, device=target.device), target.shape[0])
        aug_target = target.repeat(self.num_consts)

        # forward pass and loss computation
        logits, gates, cond = self.model(aug_input)
        to_loss, ce_loss, gt_loss = self.hparams['criterion'](logits, aug_target, gates, cons_target, self.total_filters)

        if self.global_step == 0:                      # for debugging, store example inputs
            self.example_input_array    = aug_input
            self.example_target_array   = aug_target
            self.example_cons_target    = cons_target

        # Logging to TensorBoard by default
        if self.hparams['logging']:
            self.log('trn_to_loss', to_loss)
            self.log('trn_ce_loss', ce_loss)
            self.log('trn_gt_loss', gt_loss)

            # compute layer wise on-gates
            for i, g in enumerate(gates):
                avg_on_gates = torch.sum(torch.reshape(torch.sum(g, dim=1), (self.num_consts, input.shape[0])), dim=1)/input.shape[0]
                log_cons_on_gates = {}
                for j, con in enumerate(self.hparams['constraints']):
                    log_cons_on_gates[str(con)] = avg_on_gates[j].item()
                self.log('trn_gts_lyr'+str(i), log_cons_on_gates, logger=True, on_step=True, on_epoch=False)

            # compute total accuracy
            self.log('trn_acc', self.accuracy(logits, aug_target), logger=True, on_step=True, on_epoch=False)
            # compute constraint-wise accuracy
            log_acc_gts = {}
            pred_splits = torch.split(logits, input.shape[0])
            for i, con in enumerate(self.hparams['constraints']):
                log_acc_gts['trn_acc_gts'+str(con)] = self.accuracy(pred_splits[i], target)
            self.log('trn_acc_gts', log_acc_gts, logger=True, on_step=True, on_epoch=False)

        return to_loss

    def validation_step(self, batch, batch_idx):
        input, target = batch

        # augmenting dataset here, very hacky, fix later do this in dataloader
        aug_input = input.repeat(self.num_consts, 1, 1, 1)
        cons = torch.repeat_interleave(torch.tensor(self.norm_consts, device=aug_input.device), input.shape[0]).repeat(input.shape[2], input.shape[3], 1, 1).T
        aug_input = torch.cat((aug_input, cons), 1)
        cons_target = torch.repeat_interleave(torch.tensor(self.norm_consts, device=target.device), target.shape[0])
        aug_target = target.repeat(self.num_consts)

        # forward pass and loss computation
        logits, gates, cond = self.model(aug_input)
        to_loss, ce_loss, gt_loss = self.hparams['criterion'](logits, aug_target, gates, cons_target, self.total_filters)

        # Logging to TensorBoard by default
        if self.hparams['logging']:
            self.log('val_to_loss', to_loss)
            self.log('val_ce_loss', ce_loss)
            self.log('val_gt_loss', gt_loss)

            # compute layer wise on-gates
            for i, g in enumerate(gates):
                avg_on_gates = torch.sum(torch.reshape(torch.sum(g, dim=1), (self.num_consts, input.shape[0])), dim=1)/input.shape[0]
                log_cons_on_gates = {}
                for j, con in enumerate(self.hparams['constraints']):
                    log_cons_on_gates[str(con)] = avg_on_gates[j].item()
                self.log('val_gts_lyr'+str(i), log_cons_on_gates, logger=True, on_step=False, on_epoch=True)

            # compute total accuracy
            self.log('val_acc', self.accuracy(logits, aug_target), logger=True, on_step=False, on_epoch=True)
            # compute constraint-wise accuracy
            log_acc_gts = {}
            pred_splits = torch.split(logits, input.shape[0])
            for i, con in enumerate(self.hparams['constraints']):
                log_acc_gts[str(con)] = self.accuracy(pred_splits[i], target)
            self.log('val_acc_gts', log_acc_gts, logger=True, on_step=False, on_epoch=True)

        return to_loss

    def on_validation_epoch_end(self):
        # Logging to TensorBoard by default
        if self.hparams['logging']:
            if self.current_epoch == 0:
                self.logger.experiment.add_graph(GatedCNN(self.hparams).to(self.example_input_array.device),
                                                 self.example_input_array[0, :][None, :, :, :].clone().detach()
                                                 )

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # Logging to TensorBoard by default
        if self.hparams['logging']:
            if self.global_step % 500 == 0:
                # at the end of every epoch, check what gradients w.r.t to each loss function were
                if self.example_input_array.grad is not None:
                    self.example_input_array.grad.detach_()
                    self.example_input_array.grad.zero_()
                if self.example_target_array.grad is not None:
                    self.example_target_array.grad.detach_()
                    self.example_target_array.grad.zero_()
                if self.example_cons_target.grad is not None:
                    self.example_cons_target.grad.detach_()
                    self.example_cons_target.grad.zero_()

                opt = self.optimizers(use_pl_optimizer=False)
                opt.zero_grad()

                logits, gates, cond = self.model(self.example_input_array)
                to_loss, ce_loss, gt_loss = self.hparams['criterion'](logits, self.example_target_array, gates, self.example_cons_target, self.total_filters)
                ce_loss.backward()

                for k, v in self.named_parameters():
                    if 'bn' not in k:
                        if v.grad is not None:
                            self.logger.experiment.add_histogram(
                                tag='ce.'+k+'.grad', values=v.grad, global_step=self.global_step
                            )

                opt.zero_grad()  # dont need to call zero_grad again, lightning will call it again before train_step

                if not self.hparams['man_gates']:
                    if self.example_input_array.grad is not None:
                        self.example_input_array.grad.detach_()
                        self.example_input_array.grad.zero_()
                    if self.example_target_array.grad is not None:
                        self.example_target_array.grad.detach_()
                        self.example_target_array.grad.zero_()
                    if self.example_cons_target.grad is not None:
                        self.example_cons_target.grad.detach_()
                        self.example_cons_target.grad.zero_()

                    opt = self.optimizers(use_pl_optimizer=False)
                    opt.zero_grad()

                    logits, gates, cond = self.model(self.example_input_array)
                    to_loss, ce_loss, gt_loss = self.hparams['criterion'](logits, self.example_target_array, gates, self.example_cons_target, self.total_filters)
                    gt_loss.backward()

                    for k, v in self.named_parameters():
                        if 'bn' not in k:
                            if v.grad is not None:
                                self.logger.experiment.add_histogram(
                                    tag='gt.'+k+'.grad', values=v.grad, global_step=self.global_step
                                )

                    opt.zero_grad()  # dont need to call zero_grad again, lightning will call it again before train_step

    def get_activation(self, name):
        def hook(model, input, output):                 # this will be called on every training step
            #log stuff here, selg.llog is available
            # print('Here')
            # if v.grad is not None:
            if self.hparams['logging']:
                if self.global_step % 500 == 0:
                    self.logger.experiment.add_histogram(
                        tag=name, values=output, global_step=self.global_step
                    )
        return hook

