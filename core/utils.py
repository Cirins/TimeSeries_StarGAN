from os.path import join as ospj
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of {}: {:,}".format(name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)



def save_time_series(data, labels, ncol, filename, class_names, channel_names):
    N = data.size(0)
    nrow = (N + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 2.5))
    axs = axs.flatten()
    for idx in range(N):
        for i in range(data.size(1)):
            axs[idx].plot(data[idx, i, :].cpu().numpy(), label=channel_names[i], linewidth=0.7)
        axs[idx].set_ylim(0, 1)
        axs[idx].axis('off')
        axs[idx].set_title(f'{class_names[labels[idx].item()]}')
        if idx < ncol:
            axs[idx].legend(loc='lower left')
    for idx in range(N, len(axs)):
        axs[idx].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_pickle(data, labels, save_dir, filename):
    with open(ospj(save_dir, filename), 'wb') as f:
        pickle.dump((data, labels), f)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    x_fake = nets.generator(x_src, s_ref)
    s_src = nets.style_encoder(x_src, y_src)
    x_rec = nets.generator(x_fake, s_src)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    y_concat = torch.cat([y_src, y_ref, y_ref, y_src], dim=0)
    # save_pickle(x_concat, y_concat, args.sample_dir, 'cycle_consistency.pkl')
    save_time_series(x_concat, y_concat, N, filename, args.class_names, args.channel_names)
    del x_concat
    del y_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_src, y_trg_list, z_trg_list, filename):
    N, C, W = x_src.size()
    x_concat = [x_src]
    y_concat = [y_src]

    for i, y_trg in enumerate(y_trg_list):
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]
            y_concat += [y_trg]

    x_concat = torch.cat(x_concat, dim=0)
    y_concat = torch.cat(y_concat, dim=0)
    # save_pickle(x_concat, y_concat, args.sample_dir, 'latent.pkl')
    save_time_series(x_concat, y_concat, N, filename, args.class_names, args.channel_names)
    del x_concat
    del y_concat


@torch.no_grad()
def translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, W = x_src.size()

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src]
    y_concat = [y_src]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
        y_concat += [y_ref[i:i+1], y_src]

    x_concat = torch.cat(x_concat, dim=0)
    y_concat = torch.cat(y_concat, dim=0)
    save_time_series(x_concat, y_concat, N, filename, args.class_names, args.channel_names)
    del x_concat
    del y_concat


@torch.no_grad()
def debug_sample(nets, args, inputs, step):
    print('Saving debug samples...')

    start = time.time()

    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided sample synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in range(args.num_classes)]
    z_trg_list = torch.randn(1, 1, args.latent_dim).repeat(1, N, 1).to(device)
    filename = ospj(args.sample_dir, '%06d_latent.jpg' % (step))
    translate_using_latent(nets, args, x_src, y_src, y_trg_list, z_trg_list, filename)

    # # reference-guided sample synthesis
    # filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    # translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename)

    print('Elapsed time: %f' % (time.time() - start))


