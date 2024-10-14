import os
import csv
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from statsmodels.regression.linear_model import burg
from sklearn.model_selection import KFold
import lightgbm as lgb
import sys

from core.scores.domain_score import DomainClassifier, calculate_domain_scores, save_domain_scores
from core.scores.dist_score import SiameseNet, calculate_dist_scores, save_dist_scores
from core.scores.tstr_score import calculate_classification_scores


def get_data(dataset_name, class_idx, num_train_domains):

    # Load the dataset
    with open(f'data/{dataset_name}.pkl', 'rb') as f:
        x, y, k = pickle.load(f)

    with open(f'data/{dataset_name}_fs.pkl', 'rb') as f:
        fs = pickle.load(f)

    # Filter out the samples that are used for finetuning
    x = x[fs == 0]
    y = y[fs == 0]
    k = k[fs == 0]
    
    x_ = x[(y == class_idx) & (k >= num_train_domains)]
    y_ = y[(y == class_idx) & (k >= num_train_domains)]
    k_ = k[(y == class_idx) & (k >= num_train_domains)] - num_train_domains

    return x_, y_, k_


@torch.no_grad()
def calculate_metrics(nets, args, step, mode='latent'):
    print('\nCalculating evaluation metrics in %s mode...\n' % mode)

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_classifier_te = DomainClassifier(args.num_timesteps, args.num_channels, args.num_test_domains, args.num_classes)
    filename = f'pretrained_nets/domain_classifier_{args.dataset}_te.ckpt'
    domain_classifier_te.load_state_dict(torch.load(filename, map_location=device, weights_only=False))
    domain_classifier_te = domain_classifier_te.to(device)

    siamese_net_te = SiameseNet(args.num_channels, args.num_classes, args.num_timesteps)
    filename = f'pretrained_nets/siamese_net_{args.dataset}_te.ckpt'
    siamese_net_te.load_state_dict(torch.load(filename, map_location=device, weights_only=False))
    siamese_net_te = siamese_net_te.to(device)

    classes_dict = {clss: i for i, clss in enumerate(args.class_names)}
    
    for src_class in args.class_names:

        src_idx = classes_dict[src_class]
        x_src, y_src, k_src = get_data(args.dataset_name, src_idx, args.num_train_domains)

        x_src = torch.tensor(x_src, dtype=torch.float32).to(device)

        N = len(x_src)
        
        trg_classes = [clss for clss in args.class_names if clss != src_class]

        syn_data = []
        syn_labels = []
        syn_doms = []

        for trg_class in trg_classes:

            trg_idx = classes_dict[trg_class]
            y_trg = torch.tensor([trg_idx] * N).to(device)

            if mode == 'latent':
                z_trg = torch.randn(N, args.latent_dim).to(device)
                s_trg = nets.mapping_network(z_trg, y_trg)
            else: # mode == 'reference'
                x_ref = get_data(args.dataset_name, trg_idx, args.num_train_domains)[0]
                N2 = len(x_ref)
                replace = N2 < N
                idx = np.random.choice(N2, N, replace=replace)
                x_ref = torch.tensor(x_ref[idx], dtype=torch.float32).to(device)
                s_trg = nets.style_encoder(x_ref, y_trg)

            x_fake = nets.generator(x_src, s_trg)
            k_fake = k_src.copy()

            domain_scores = calculate_domain_scores(domain_classifier_te, x_fake, y_trg, k_fake, src_class, trg_class)
            save_domain_scores(domain_scores, src_class, trg_class, step, mode, args.eval_dir)

            dist_scores = calculate_dist_scores(siamese_net_te, x_fake, y_trg, k_fake, src_class, trg_class, 
                                                args.dataset_name, args.num_train_domains, args.class_names)
            save_dist_scores(dist_scores, src_class, trg_class, step, mode, args.eval_dir)

            syn_data.append(x_fake)
            syn_labels.append(y_trg)
            syn_doms.append(k_fake)

        syn_data = torch.cat(syn_data, dim=0)
        syn_labels = torch.cat(syn_labels, dim=0)
        syn_doms = np.concatenate(syn_doms, axis=0)

        calculate_classification_scores(syn_data, syn_labels, syn_doms, src_class, trg_classes, step, mode, 
                                        args.eval_dir, args.class_names, args.dataset_name, args.num_train_domains)

    print('Total time taken:', time.time() - start_time, '\n')


