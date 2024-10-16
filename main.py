import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

import numpy as np

import datetime

from core.data_loader import get_train_loader, get_eval_loader
from core.solver import Solver

import multiprocessing as mp
mp.set_start_method('spawn', force=True)



def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        assert args.loss_type in ['minimax', 'wgan', 'lsgan']
        assert args.n_critic == 1 or args.loss_type == 'wgan'
        loaders = Munch(src=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names,
                                             num_train_domains=args.num_train_domains,
                                             which='source', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers),
                        ref=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names, 
                                             num_train_domains=args.num_train_domains,
                                             which='reference', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers),
                        val=get_eval_loader(dataset_name=args.dataset_name,  
                                            class_names=args.class_names,
                                            num_train_domains=args.num_train_domains,
                                            batch_size=args.val_batch_size, 
                                            num_workers=args.num_workers))
        solver.train(loaders)
    
    elif args.mode == 'finetune':
        assert args.loss_type in ['minimax', 'wgan', 'lsgan']
        assert args.n_critic == 1 or args.loss_type == 'wgan'
        loaders = Munch(src=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names,
                                             num_train_domains=args.num_train_domains,
                                             which='source', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             finetune=True),
                        ref=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names, 
                                             num_train_domains=args.num_train_domains,
                                             which='reference', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             finetune=True),
                        val=get_eval_loader(dataset_name=args.dataset_name,  
                                            class_names=args.class_names,
                                            num_train_domains=args.num_train_domains,
                                            batch_size=args.val_batch_size, 
                                            num_workers=args.num_workers))
        solver.train(loaders)

    elif args.mode == 'sample':
        solver.sample()

    elif args.mode == 'eval':
        assert args.loss_type in ['minimax', 'wgan', 'lsgan']
        assert args.n_critic == 1 or args.loss_type == 'wgan'
        loaders = Munch(src=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names,
                                             num_train_domains=args.num_train_domains,
                                             which='source', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers),
                        ref=get_train_loader(dataset_name=args.dataset_name, 
                                             class_names=args.class_names, 
                                             num_train_domains=args.num_train_domains,
                                             which='reference', 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers),
                        val=get_eval_loader(dataset_name=args.dataset_name,  
                                            class_names=args.class_names,
                                            num_train_domains=args.num_train_domains,
                                            batch_size=args.val_batch_size, 
                                            num_workers=args.num_workers))
        solver.evaluate(loaders)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='realworld',
                        choices=['realworld', 'cwru'], help='Dataset')
    parser.add_argument('--dataset_name', type=str, default='realworld_128_3ch_4cl',
                        choices=['realworld_128_3ch_4cl', 'cwru_256_3ch_5cl'], help='Dataset name')
    # parser.add_argument('--class_names', type=str, nargs='+', default=['WAL', 'RUN', 'CLD', 'CLU'],
    #                     help='Class names for dataset')
    parser.add_argument('--class_names', type=str, required=True)
    # parser.add_argument('--channel_names', type=str, nargs='+', default=['X', 'Y', 'Z'],
    #                     help='Channel names for dataset')
    parser.add_argument('--channel_names', type=str, required=True)

    # model arguments
    parser.add_argument('--num_timesteps', type=int, default=128,
                        help='Number of timesteps')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='Number of channels')
    parser.add_argument('--num_train_domains', type=int, default=10,
                        help='Number of train domains')
    parser.add_argument('--num_test_domains', type=int, default=5,
                        help='Number of test domains')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--max_conv_dim', type=int, default=512,
                        help='Maximum number of channels')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_id', type=float, default=1,
                        help='Weight for identity loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_dom', type=float, default=0,
                        help='Weight for domain classification loss')
    parser.add_argument('--lambda_trts', type=float, default=0,
                        help='Weight for TRTS loss')
    parser.add_argument('--lambda_ds', type=float, default=0,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='Weight for gradient penalty')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--dom_iter', type=int, default=0,
                        help='Number of iterations to optimize domain loss')
    parser.add_argument('--dom_start', type=int, default=0,
                        help='Number of iterations to start domain loss')
    parser.add_argument('--trts_iter', type=int, default=0,
                        help='Number of iterations to optimize trts loss')
    parser.add_argument('--trts_start', type=int, default=0,
                        help='Number of iterations to start trts loss')

    # training arguments
    parser.add_argument('--loss_type', type=str, default='lsgan',
                        help='Loss function type [minimax | wgan | lsgan]')
    parser.add_argument('--n_critic', type=int, default=1,
                        help='Number of critic iterations per generator iteration')
    parser.add_argument('--total_iters', type=int, default=500000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'finetune'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=2710,
                        help='Seed for random number generator')
    parser.add_argument('--num_syn', type=int, default=1,
                        help='Number of synthetic datasets to generate')

    # directory for training
    # parser.add_argument('--sample_dir', type=str, default='expr/samples',
    #                     help='Directory for saving generated samples')
    # parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
    #                     help='Directory for saving network checkpoints')
    # parser.add_argument('--history_dir', type=str, default='expr/history',
    #                     help='Directory for saving training history')
    # parser.add_argument('--eval_dir', type=str, default=f'expr/eval',
    #                     help='Directory for saving metrics')
    # parser.add_argument('--syn_dir', type=str, default='expr/syn_datasets',
    #                     help='Directory for saving synthetic samples')

    # step size
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=2000)

    args = parser.parse_args()
    args.class_names = args.class_names.split(',')
    args.channel_names = args.channel_names.split(',')

    args.expr_dir = f'expr_{args.dataset}'
    args.sample_dir = os.path.join(args.expr_dir, 'samples')
    args.checkpoint_dir = os.path.join(args.expr_dir, 'checkpoints')
    args.history_dir = os.path.join(args.expr_dir, 'history')
    args.eval_dir = os.path.join(args.expr_dir, 'eval')
    args.syn_dir = os.path.join(args.expr_dir, 'syn_datasets')

    main(args)
