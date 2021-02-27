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
from torch import nn
from model.vanilla_conv import VanillaCNN


class LightningVanillaCNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = VanillaCNN()
        self.total_filters = 13.0   # make this dynamic
        self.norm_consts = [float(g)/self.total_filters for g in self.hparams['constraints']]
        self.num_consts = len(self.hparams['constraints'])
        self.accuracy = pl.metrics.Accuracy()

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
        logits = self.model(aug_input)
        ce_loss = self.hparams['criterion'](logits, aug_target)

        if self.global_step == 0:                      # for debugging, store example inputs
            self.example_input_array    = aug_input
            self.example_target_array   = aug_target
            self.example_cons_target    = cons_target

        # Logging to TensorBoard by default
        if self.hparams['logging']:
            self.log('trn_ce_loss', ce_loss)

            # compute total accuracy
            self.log('trn_acc', self.accuracy(logits, aug_target), logger=True, on_step=True, on_epoch=False)
            # compute constraint-wise accuracy
            log_acc_gts = {}
            pred_splits = torch.split(logits, input.shape[0])
            for i, con in enumerate(self.hparams['constraints']):
                log_acc_gts['trn_acc_gts'+str(con)] = self.accuracy(pred_splits[i], target)
            self.log('trn_acc_gts', log_acc_gts, logger=True, on_step=True, on_epoch=False)

        return ce_loss

    def validation_step(self, batch, batch_idx):
        input, target = batch

        # augmenting dataset here, very hacky, fix later do this in dataloader
        aug_input = input.repeat(self.num_consts, 1, 1, 1)
        cons = torch.repeat_interleave(torch.tensor(self.norm_consts, device=aug_input.device), input.shape[0]).repeat(input.shape[2], input.shape[3], 1, 1).T
        aug_input = torch.cat((aug_input, cons), 1)
        cons_target = torch.repeat_interleave(torch.tensor(self.norm_consts, device=target.device), target.shape[0])
        aug_target = target.repeat(self.num_consts)

        # forward pass and loss computation
        logits = self.model(aug_input)
        ce_loss = self.hparams['criterion'](logits, aug_target)

        # Logging to TensorBoard by default
        if self.hparams['logging']:
            self.log('val_ce_loss', ce_loss)

            # compute total accuracy
            self.log('val_acc', self.accuracy(logits, aug_target), logger=True, on_step=False, on_epoch=True)
            # compute constraint-wise accuracy
            log_acc_gts = {}
            pred_splits = torch.split(logits, input.shape[0])
            for i, con in enumerate(self.hparams['constraints']):
                log_acc_gts[str(con)] = self.accuracy(pred_splits[i], target)
            self.log('val_acc_gts', log_acc_gts, logger=True, on_step=False, on_epoch=True)

        return ce_loss

    def on_after_backward(self):
        # Logging to TensorBoard by default
        if self.hparams['logging']:
            if self.trainer.global_step % 500 == 0:
                for k, v in self.named_parameters():
                    if 'conv' in k:
                        for i in range(v.grad.shape[0]):     # for every filter in the conv layer
                            self.logger.experiment.add_histogram(
                                tag=k+'.filt'+str(i), values=v.grad[i], global_step=self.trainer.global_step
                            )   # log every filters gradients individually
                    elif 'bn' in k:
                        continue
                    else:
                        self.logger.experiment.add_histogram(
                            tag=k, values=v.grad, global_step=self.trainer.global_step
                        )

    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     # Logging to TensorBoard by default
    #     if self.hparams['logging']:
    #         if self.global_step % 500 == 0:
    #             # at the end of every epoch, check what gradients w.r.t to each loss function were
    #             if self.example_input_array.grad is not None:
    #                 self.example_input_array.grad.detach_()
    #                 self.example_input_array.grad.zero_()
    #             if self.example_target_array.grad is not None:
    #                 self.example_target_array.grad.detach_()
    #                 self.example_target_array.grad.zero_()
    #             if self.example_cons_target.grad is not None:
    #                 self.example_cons_target.grad.detach_()
    #                 self.example_cons_target.grad.zero_()
    #
    #             opt = self.optimizers(use_pl_optimizer=False)
    #             opt.zero_grad()
    #
    #             logits = self.model(self.example_input_array)
    #             ce_loss = self.hparams['criterion'](logits, self.example_target_array)
    #             ce_loss.backward()
    #
    #             for k, v in self.named_parameters():
    #                 if v.grad is not None:
    #                     self.logger.experiment.add_histogram(
    #                         tag='ce.'+k+'.grad', values=v.grad, global_step=self.global_step
    #                     )
    #
    #             opt.zero_grad()  # dont need to call zero_grad again, lightning will call it again before train_step

