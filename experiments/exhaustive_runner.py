import os
from multiprocessing import Pool
import argparse
import itertools

# python H:\mycode\GatedNetsDir\GatedNets\exhaustive_runner.py
# conda activate H:\mycode\my_envs\py3.6_torch1.7.1_


def run_process(process):
    print(process)
    # os.system('python {}'.format(process))
    os.system(str)


def main(args):

    print('Try something funky ...')

    assert args.subnet_constraint > len(args.arch_geno)

    geno_edges = []
    for f in args.arch_geno:
        geno_edges.append(list(range(1, f+1)))
    print(geno_edges)

    procs = []
    subnet_genos = list(itertools.product(*geno_edges))
    for subnet_geno in subnet_genos:
        if sum(subnet_geno) == args.subnet_constraint:
            man_on_gates = ''
            for a in subnet_geno:
                man_on_gates = man_on_gates + str(a) + ' '
            comms = []
            for i in range(args.num_runs):
                test_args = f' --env {args.env}' \
                       f' --data {args.data}' \
                       f' --epochs {args.epochs}' \
                       f' --batch_size {args.batch_size}' \
                       f' --learning_rate {args.learning_rate}' \
                       f' --weight_decay {args.weight_decay}' \
                       f' --momentum {args.momentum}' \
                       f' --gate_loss_weight {args.gate_loss_weight}' \
                       f' --gate_loss {args.gate_loss}' \
                       f' --criterion {args.criterion}' \
                       f' --constraints {args.subnet_constraint}' \
                       ' --man_gates True' \
                       f' --man_on_gates {man_on_gates}' \
                       f' --num_gpus {i}' \
                       f' --logging {args.logging}' \
                       f' --num_workers {args.num_workers}'
                comm = 'python ' + args.prefix + 'lightning_train.py' + test_args
                comms.append(comm)
            procs.append(comms)

    # for i in range()


    pool = Pool(processes=4)
    for p in procs:                 # run 4 at a time
        pool.map(run_process, p)


    # pool.map(run_process, proc_1)
    # pool.map(run_process, proc_2)

    # os.system('python H:\mycode\GatedNetsDir\GatedNets\sandbox.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='H:\mycode\GatedNetsDir\GatedNets\\')
    parser.add_argument('--arch_geno', type=int, nargs='+', default=[3, 10], help='genotype of the arch in which to exhaustively train subnets')
    parser.add_argument('--subnet_constraint', type=int, default=4, help='constraint on subnet')
    parser.add_argument('--num_runs', type=int, default=4, help='number of runs for statistical significance')
    parser.add_argument('--env', type=str, default='ecs_gpu', help='environment used')
    parser.add_argument('--data', type=str, default='H:\\mycode\\data', help='dirctory for data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--gate_loss_weight', type=float, default=0.0, help='gate loss weight in loss function')
    parser.add_argument('--gate_loss', type=str, default='l2', help='loss function used in gates loss')
    parser.add_argument('--criterion', type=int, default=0, help='multi-objective criterion. 0-PerformanceLoss')
    # parser.add_argument('--constraints', type=int, nargs='+', default=[4], help='number of constraints used in training')
    # parser.add_argument('--man_gates', type=bool, default=True, help='use manual gating')
    # parser.add_argument('--man_on_gates', type=int, nargs='+', default=[3, 5, 7], help='number of on gates in each layer')
    parser.add_argument('--num_gpus', type=int, nargs='+', default=[0], help='number of gpus in training')
    parser.add_argument('--logging', type=bool, default=True, help='turn on/off logging')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers in dataloader')
    args = parser.parse_args()
    main(args)
