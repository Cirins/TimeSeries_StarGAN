import time
import torch
import pickle
import numpy as np


def get_data(dataset_name, class_idx, num_train_domains):

    # Load the dataset
    with open(f'data/{dataset_name}.pkl', 'rb') as f:
        x, y, k = pickle.load(f)
    
    x_ = x[(y == class_idx) & (k >= num_train_domains)]
    y_ = y[(y == class_idx) & (k >= num_train_domains)]
    k_ = k[(y == class_idx) & (k >= num_train_domains)]

    return x_, y_, k_



@torch.no_grad()
def sample_timeseries(nets, args, mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_dict = {clss: i for i, clss in enumerate(args.class_names)}

    syn_data = {}
    
    for src_class in args.class_names:

        src_idx = classes_dict[src_class]
        x_src, y_src, k_src = get_data(args.dataset_name, src_idx, args.num_train_domains)

        x_src = torch.tensor(x_src, dtype=torch.float32).to(device)

        N = len(x_src)
        
        trg_classes = [x for x in args.class_names if x != src_class]

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
            k_fake = k_src

            syn_data[(src_class, trg_class)] = (x_fake.cpu().detach().numpy(), y_trg.cpu().detach().numpy(), k_fake)

            print(f'Generated {src_class} -> {trg_class} ({N} samples)')

    return syn_data


def sample_all(nets, args):
    start_time = time.time()
    
    for i in range(args.num_syn):
        print(f'Generating samples in latent mode ({i+1}/{args.num_syn})')
        syn_data = sample_timeseries(nets, args, mode='latent')
        with open(f'{args.syn_dir}/syndata_lat_{i+1}.pkl', 'wb') as f:
            pickle.dump(syn_data, f)

        print(f'Generating samples in reference mode ({i+1}/{args.num_syn})')
        syn_data = sample_timeseries(nets, args, mode='reference')
        with open(f'{args.syn_dir}/syndata_ref_{i+1}.pkl', 'wb') as f:
            pickle.dump(syn_data, f)

    print(f'Finished generating samples in {time.time() - start_time:.2f} seconds')





