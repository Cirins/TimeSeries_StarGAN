import os
from os.path import join as ospj
import time
import datetime
import csv
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from core.eval import calculate_metrics
from core.sampler import sample_all


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on', self.device)

        self.nets, self.nets_ema, self.domain_classifier_df, self.trts_classifier_tr = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        setattr(self, 'domain_classifier_df', self.domain_classifier_df)
        utils.print_network(self.domain_classifier_df, 'domain_classifier_df')
        setattr(self, 'trts_classifier_tr', self.trts_classifier_tr)
        utils.print_network(self.trts_classifier_tr, 'trts_classifier_tr')
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        # Load the pretrained domain classifier
        print('Loading the pretrained domain classifier...')
        self.domain_classifier_df.load_state_dict(torch.load(f'pretrained_nets/domain_classifier_{args.dataset}_df.ckpt', map_location=self.device))
        self.domain_classifier_df.eval()

        # # Load the pretrained trts classifier
        # print('Loading the pretrained trts classifier...')
        # self.trts_classifier_tr.load_state_dict(torch.load('pretrained_nets/trts_classifier_tr.ckpt', map_location=self.device))
        # self.trts_classifier_tr.eval()

        if args.mode == 'train' or args.mode == 'finetune':
            self.optims = Munch()
            for net in self.nets.keys():
                print('Setting up optimizer for %s...' % net)
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(args.checkpoint_dir, '{:06d}_nets.ckpt', data_parallel=True, **self.nets),
                CheckpointIO(args.checkpoint_dir, '{:06d}_nets_ema.ckpt', data_parallel=True, **self.nets_ema),
                CheckpointIO(args.checkpoint_dir, '{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [CheckpointIO(args.checkpoint_dir, '{:06d}_nets_ema.ckpt', data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            if ('ema' not in name) and ('domain_classifier_df' not in name) and ('trts_classifier_tr' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        domain_classifier_df = self.domain_classifier_df
        trts_classifier_tr = self.trts_classifier_tr
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # # calculate lambda_ds decay per iteration
        # initial_lambda_ds = args.lambda_ds
        # lambda_ds_decay = initial_lambda_ds / args.total_iters

        # # calculate lambda_dom increment per iteration
        # if args.dom_iter > 0:
        #     final_lambda_dom = args.lambda_dom
        #     lambda_dom_increment = final_lambda_dom / args.dom_iter

        # # calculate lambda_trts increment per iteration
        # if args.trts_iter > 0:
        #     final_lambda_trts = args.lambda_trts
        #     lambda_trts_increment = final_lambda_trts / args.trts_iter

        start_time = time.time()

        print('Start training...')
        for i in range(args.resume_iter, args.total_iters):
            # fetch samples and labels
            inputs = next(fetcher)
            x_real, y_org, k_org = inputs.x_src, inputs.y_src, inputs.k_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg,
                                                     z_trg=z_trg, loss_type=args.loss_type)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, 
                                                  x_ref=x_ref, loss_type=args.loss_type)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            if i % args.n_critic == 0:
                g_loss, g_losses_latent = compute_g_loss(nets, i, args, domain_classifier_df, trts_classifier_tr,
                                                         x_real, y_org, y_trg, k_org, z_trgs=[z_trg, z_trg2], loss_type=args.loss_type)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()

                g_loss, g_losses_ref = compute_g_loss(nets, i, args, domain_classifier_df, trts_classifier_tr,
                                                      x_real, y_org, y_trg, k_org, x_refs=[x_ref, x_ref2], loss_type=args.loss_type)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # # decay weight for diversity sensitive loss
            # if args.lambda_ds > 0:
            #     args.lambda_ds = max(0, initial_lambda_ds - (i+1)*lambda_ds_decay)

            # # increment weight for domain classification loss
            # if i < args.dom_start:
            #     args.lambda_dom = 0
            # elif args.dom_start <= i < args.dom_start + args.dom_iter:
            #     args.lambda_dom = (i-args.dom_start)*lambda_dom_increment
            # else:
            #     args.lambda_dom = final_lambda_dom if args.dom_iter > 0 else 0

            # # increment weight for TRTS loss
            # if i < args.trts_start:
            #     args.lambda_trts = 0
            # elif args.trts_start <= i < args.trts_start + args.trts_iter:
            #     args.lambda_trts = (i-args.trts_start)*lambda_trts_increment
            # else:
            #     args.lambda_trts = final_lambda_trts if args.trts_iter > 0 else 0

            # print out log info
            if (i+1) % args.print_every == 0:
                history_dict = get_history(i, args, start_time, d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref)
                print_log(history_dict)
                save_history(history_dict, args.history_dir)

            # generate samples for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_sample(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute evaluation metrics
            if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.syn_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        sample_all(nets_ema, args)

    @torch.no_grad()
    def evaluate(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, loss_type='minimax'):
    assert (z_trg is None) != (x_ref is None)
    # with real samples
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1, loss_type=loss_type)
    loss_reg = r1_reg(out, x_real)

    # with fake samples
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0, loss_type=loss_type)
    
    # Compute gradient penalty if using Wasserstein loss with gradient penalty
    if loss_type == 'wgan':
        gp = compute_gradient_penalty(nets.discriminator, x_real, x_fake, y_trg)
        loss = loss_real + loss_fake + args.lambda_reg * loss_reg + args.lambda_gp * gp
        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item(),
                           gp=gp.item())
    else:
        loss = loss_real + loss_fake + args.lambda_reg * loss_reg
        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())


def compute_g_loss(nets, step, args, domain_classifier_df, trts_classifier_tr,
                   x_real, y_org, y_trg, k_org, z_trgs=None, x_refs=None, loss_type='minimax'):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else: # x_refs is not None
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1, loss_type=loss_type)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else: # x_refs is not None
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # identity loss
    x_id = nets.generator(x_real, s_org)
    loss_id = torch.mean(torch.abs(x_id - x_real))

    # domain loss
    k_fake = domain_classifier_df(x_fake, y_trg)
    loss_dom = nn.CrossEntropyLoss()(k_fake, k_org)

    # # TRTS loss
    # y_fake = trts_classifier_tr(x_fake)
    # loss_trts = nn.CrossEntropyLoss()(y_fake, y_trg)

    # total loss
    loss = loss_adv + args.lambda_cyc * loss_cyc + args.lambda_id * loss_id + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_dom * loss_dom

    return loss, Munch(adv=loss_adv.item(),
                       dom=loss_dom.item(),
                    #    trts=loss_trts.item(),
                       cyc=loss_cyc.item(),
                       id=loss_id.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item()
                       )


def compute_gradient_penalty(discriminator, real_data, fake_data, y_trg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Random weight term for interpolation between real and fake data
    alpha = torch.rand(real_data.size(0), 1, 1).expand_as(real_data).to(device)
    
    # Interpolated data
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    
    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated, y_trg)
    
    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # Gradients have shape (batch_size, num_channels, sequence_length), flatten to (batch_size, -1)
    gradients = gradients.view(gradients.size(0), -1)
    
    # Derive the penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target, loss_type='minimax'):
    assert target in [1, 0]
    if loss_type == 'minimax':
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
    elif loss_type == 'wgan':
        if target == 1:
            loss = -torch.mean(logits)
        else:
            loss = torch.mean(logits)
    elif loss_type == 'lsgan':
        if target == 1:
            loss = torch.mean((logits - 1) ** 2) * 0.5
        else:
            loss = torch.mean((logits) ** 2) * 0.5
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real samples
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def get_history(i, args, start_time, d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref):
    elapsed = time.time() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    
    all_losses = {}
    for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
        for key, value in loss.items():
            all_losses[prefix + key] = value
    all_losses['G/lambda_ds'] = args.lambda_ds
    all_losses['G/lambda_dom'] = args.lambda_dom
    all_losses['G/lambda_trts'] = args.lambda_trts
    
    history_dict = {
        'iteration': i + 1,
        'total_iters': args.total_iters,
        'elapsed': elapsed_str,
        'losses': all_losses
    }
    return history_dict


def print_log(history_dict):
    log = "Elapsed time [{}], Iteration [{}/{}], ".format(history_dict['elapsed'], history_dict['iteration'], history_dict['total_iters'])
    log += ' '.join(['{}: [{:.4f}]'.format(key, value) for key, value in history_dict['losses'].items()])
    print(log)


def save_history(history_dict, history_dir):
    os.makedirs(history_dir, exist_ok=True)

    # Define the CSV file path
    csv_path = os.path.join(history_dir, 'training_history.csv')
    
    # Check if the file exists to decide on writing headers
    file_exists = os.path.isfile(csv_path)
    
    # Open the file in append mode, create if does not exist
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file does not exist, write the header
        if not file_exists:
            headers = ['Elapsed Time', 'Iteration', 'Total Iterations'] + [key for key in history_dict['losses'].keys()]
            writer.writerow(headers)
        
        # Prepare data row, formatting loss values to six decimal places
        data = [history_dict['elapsed'], history_dict['iteration'], history_dict['total_iters']]
        data += ['{:.6f}'.format(history_dict['losses'][key]) for key in history_dict['losses'].keys()]
        writer.writerow(data)
