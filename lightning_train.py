import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',                type=str,               default='ecs_gpu',          help='environment used')
parser.add_argument('--data',               type=str,               default='H:\\mycode\\data', help='dirctory for data')
parser.add_argument('--epochs',             type=int,               default=300,                help='number of epochs to train')
parser.add_argument('--batch_size',         type=int,               default=32,                 help='batch size')
parser.add_argument('--learning_rate',      type=float,             default=0.01,               help='init learning rate')
parser.add_argument('--learning_rate_min',  type=float,             default=0.001,              help='minimum learning rate in scheduler')
parser.add_argument('--momentum',           type=float,             default=0.9,                help='momentum')
parser.add_argument('--weight_decay',       type=float,             default=0.0,                help='weight decay')
parser.add_argument('--gate_loss_weight',   type=float,             default=0.0,                help='gate loss weight in loss function')
parser.add_argument('--gate_loss_alpha',    type=float,             default=0.4,                help='gate loss alpha')
parser.add_argument('--gate_loss_beta',     type=float,             default=1.3,                help='gate loss beta')
parser.add_argument('--criterion',          type=int,               default=0,                  help='multi-objective criterion. 0-PerformanceLoss')
parser.add_argument('--constraints',        type=int,   nargs='+',  default=[4],               help='number of constraints used in training')
parser.add_argument('--man_gates',          type=bool,              default=False,              help='use manual gating')
parser.add_argument('--man_on_gates',       type=int,   nargs='+',  default=[3, 5, 7],          help='number of on gates in each layer')
parser.add_argument('--num_gpus',           type=int,   nargs='+',  default=[0],                help='number of gpus in training')
parser.add_argument('--logging',            type=bool,              default=True,               help='turn on/off logging')
parser.add_argument('--num_workers',        type=int,               default=0,                  help='num_workers in dataloader')
args = parser.parse_args()

if args.env == 'ecs_gpu':                                           # manually add paths when using ecs_gpu
    ENV_PATHS = ["H:\\mycode\\my_envs\\py3.6_torch1.7.1_",
                 "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\mingw-w64\\bin",
                 "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\usr\\bin",
                 "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Library\\bin",
                 "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\Scripts",
                 "H:\\mycode\\my_envs\\py3.6_torch1.7.1_\\bin"]

    for path in ENV_PATHS:
        if path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + path

elif args.env == 'yann_ecs':
    ENV_PATHS = ["/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_",
                 "/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_/Library/mingw-w64/bin",
                 "/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_/Library/usr/bin",
                 "/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_/Library/bin",
                 "/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_/Script",
                 "/home/ss2n18/mydocuments/mycode/my_envs/py3.6_torch1.7.1_/bin"]

    for path in ENV_PATHS:
        if path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from model.lightning_model import LightningGatedCNN
from loss.per_loss import PerformanceLoss
from loss.per_loss import PerformanceLoss_v2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


def main():

    criterions = [PerformanceLoss(lam=args.gate_loss_weight),
                  PerformanceLoss_v2()]

    hp_criterion = criterions[args.criterion]
    hp_num_workers = args.num_workers*(len(args.num_gpus)*int(args.env != 'ecs_gpu'))
    hparams = {'epochs':            args.epochs,
               'batch_size':        args.batch_size,
               'learning_rate':     args.learning_rate,
               'learning_rate_min': args.learning_rate_min,
               'momentum':          args.momentum,
               'weight_decay':      args.weight_decay,
               'gate_loss_weight':  args.gate_loss_weight,
               'gate_loss_alpha':   args.gate_loss_alpha,
               'gate_loss_beta':    args.gate_loss_beta,
               'criterion':         hp_criterion,
               'constraints':       args.constraints,
               'num_gpus':          args.num_gpus,
               'num_workers':       hp_num_workers,
               'logging':           args.logging,
               'man_gates':         args.man_gates,
               'man_on_gates':      args.man_on_gates
               }

    print(args)
    print(hparams)

    transform = transforms.Compose([transforms.Resize(20), transforms.ToTensor()])

    trainset = CIFAR10(root=args.data, train=True, download=True, transform=transform)
    validset = CIFAR10(root=args.data, train=False, download=True, transform=transform)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=hparams['batch_size'], shuffle=True, pin_memory=True, num_workers=hp_num_workers)
    validloader = DataLoader(validset, batch_size=hparams['batch_size'], pin_memory=True, num_workers=hp_num_workers)

    lightning_model = LightningGatedCNN(hparams)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=hparams['epochs'],
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0,
                         log_every_n_steps=500,
                         callbacks=[lr_monitor]
                         )
    trainer.fit(lightning_model, trainloader, validloader)


if __name__ == '__main__':
    main()
