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

        self.automatic_optimization = False

        self.hparams = hparams
        self.gt_loss_wt = 0.0

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

            # for i in range(len(self.model.gconv3.convs)):
            #     self.model.gconv3.convs[i].register_forward_hook(self.get_activation('fwd.model.gconv3.convs'+str(i)))
            # self.model.gconv3.bn1.register_forward_hook(self.get_activation('fwd.model.gconv3.bn1'))
            # self.model.activ3.register_forward_hook(self.get_activation('fwd.model.activ3'))

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

            # self.model.gconv3.gating_nw.fc1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.fc1'))
            # # self.model.gconv3.gating_nw.bn1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.bn1'))
            # self.model.gconv3.gating_nw.activ1.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.activ1'))
            # self.model.gconv3.gating_nw.fc2.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.fc2'))
            # self.model.gconv3.gating_nw.bn2.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw.bn2'))
            # self.model.gconv3.gating_nw.register_forward_hook(self.get_activation('fwd.model.gconv3.gating_nw'))

            # self.model.gconv1.gating_nw.fc1.register_backward_hook(self.get_grads('fwd.model.gconv1.gating_nw.fc1'))

    def forward(self, x):
        out = self.model(x)

    def configure_optimizers(self):
        # optimiser = optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])
        optimiser = optim.SGD(self.parameters(), self.hparams['learning_rate'], self.hparams['momentum'], self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, float(self.hparams['epochs']), eta_min=self.hparams['learning_rate_min'])
        return [optimiser], [scheduler]

    def training_step(self, batch, batch_idx):
        input, target = batch
        opt = self.optimizers()
        if self.current_epoch < 30:
            self.gt_loss_wt = 0.0
        elif self.current_epoch < 60:
            self.gt_loss_wt = 1.0
        elif self.current_epoch < 90:
            self.gt_loss_wt = 1.5
        else:
            self.gt_loss_wt = 1.0

        self.gt_loss_wt = 0.0
        eps = 1e-9

        # augmenting dataset here, very hacky, fix later do this in dataloader
        aug_input = input.repeat(self.num_consts, 1, 1, 1)
        cons = torch.repeat_interleave(torch.tensor(self.norm_consts, device=aug_input.device), input.shape[0]).repeat(input.shape[2], input.shape[3], 1, 1).T
        aug_input = torch.cat((aug_input, cons), 1)
        cons_target = torch.repeat_interleave(torch.tensor(self.norm_consts, device=target.device), target.shape[0])
        aug_target = target.repeat(self.num_consts)

        # compute alpha for epoch - repetitive computation. shift to on_epoch_start
        next_alpha = (self.current_epoch * (self.hparams['gate_loss_alpha'] - self.hparams['gate_loss_alpha_min'])) / self.hparams['epochs'] + self.hparams['gate_loss_alpha_min']
        self.hparams['criterion'].update_alpha(next_alpha)

        # forward pass and loss computation
        logits, gates, cond = self.model(aug_input)
        to_loss, ce_loss, gt_loss = self.hparams['criterion'](logits, aug_target, gates, cons_target, self.total_filters)

        ce_grads = {}
        self.manual_backward(ce_loss, optimizer=opt, retain_graph=True)
        for name, weight in self.named_parameters():
            if weight.grad is not None:  # and 'gating_nw' in np[0]:
                ce_grads[name] = weight.grad.data.clone().detach()
                # ce_grads.append((np[0], np[1].grad.data.clone().detach()))
        # g_ce = list(self.named_parameters())[1][1].grad.data.clone().detach()
        opt.zero_grad()

        gt_grads = []
        self.manual_backward(gt_loss, optimizer=opt, retain_graph=True)
        for name, weight in self.named_parameters():
            if weight.grad is not None:  # and 'gating_nw' in np[0]:
                weight.grad = ce_grads[name] + (weight.grad/(torch.linalg.norm(weight.grad)+eps))*(self.gt_loss_wt*torch.linalg.norm(ce_grads[name]))
                # add previous ce_grad and gt_grad whose magnitude is normalised to w*|ce_grad|
                # intially hard-coding 'w' to 0.0, this implies only ce will be minimised

        # gt_grads = []
        # self.manual_backward(gt_loss, optimizer=opt, retain_graph=True)
        # for np in self.named_parameters():
        #     if np[1].grad is not None: # and 'gating_nw' in np[0]:
        #         gt_grads.append((np[0], np[1].grad.data.clone().detach()))
        # g_gt = list(self.named_parameters())[1][1].grad.data.clone().detach()
        # opt.zero_grad()

        # to_grads = []
        # self.manual_backward(to_loss, optimizer=opt, retain_graph=True)
        # for np in self.named_parameters():
        #     if np[1].grad is not None: # and 'gating_nw' in np[0]:
        #         to_grads.append((np[0], np[1].grad.data.clone().detach()))
        # g_to = list(self.named_parameters())[1][1].grad.data.clone().detach()
        # opt.zero_grad()

        # gt_grads_adj = []
        # for i, (name, gt_grad) in gt_grads:
        #     gt_grad_adj = (gt_grad/torch.linalg.norm(gt_grad))*(self.gt.loss.weight*torch.linalg.norm(ce_grads[i][1]))

        opt.step()
        opt.zero_grad()

        if self.global_step == 0:                      # for debugging, store example inputs
            self.example_input_array    = aug_input
            self.example_target_array   = aug_target
            self.example_cons_target    = cons_target

        # Logging to TensorBoard by default
        if self.hparams['logging']:
            self.log('alpha', self.hparams['criterion'].alpha)

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

        # return to_loss # manually optimising so don't need to return loss to optimiser

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
        def hook(module, input, output):                 # this will be called on every training step
            # log stuff here, selg.log is available
            # print('Here')
            # if v.grad is not None:
            if self.hparams['logging']:
                if self.global_step % 500 == 0:
                    self.logger.experiment.add_histogram(
                        tag=name, values=output, global_step=self.global_step
                    )
        return hook

    def get_grads(self, name):
        def hook(module, input, output):
            a = 1
            s = 2

            if self.hparams['logging']:
                if self.global_step % 500 == 0:
                    self.logger.experiment.add_histogram(
                        tag=name, values=output, global_step=self.global_step
                    )
        return hook