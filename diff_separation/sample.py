"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com
Description: This script performs sampling using a trained diffusion model.
It loads the trained model, processes test data, and generates samples using either DDPM or DDIM sampling.
The output samples and evaluation metrics (e.g., MSE between predicted and true velocity fields,
PDE loss values) are saved in .mat files for further analysis.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F

from code.datasets import normalizer_vp, normalizer_vs
import scipy.io as sio
from code.train_util import parse_dataname_from_filename

from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import random
import math

def main():
    # Parse command-line arguments.
    args = create_argparser().parse_args()

    # Set the device to GPU.
    device = th.device('cuda')

    # Configure logging.
    logger.configure()

    # Define the training step from which to load the model checkpoint.
    train_step = xxxxxx

    logger.log("creating model and diffusion...")
    # Create the model and diffusion process using default parameters and parsed args.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # Load the trained model state from the specified checkpoint.
    model.load_state_dict(
        th.load(f'{args.model_path}{(train_step):06d}.pt', map_location=device)
    )
    model.to(device=device)
    model.eval()  # Set the model to evaluation mode.
    criterion = th.nn.MSELoss()  # Define MSE loss for evaluation.
    logger.log("sampling...")

    # Set the number of test samples.
    data_num = 8

    # Create output directory based on whether DDIM sampling and PDE guidance are used.
    if not args.use_ddim:
        if args.pde_guide:
            dir_output = f'./output/ddpm/step{train_step}/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output/ddpm/step{train_step}/usepde_{args.pde_guide}/'
    else:
        if args.pde_guide:
            dir_output = f'./output/{args.timestep_respacing}/step{train_step}/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output/{args.timestep_respacing}/step{train_step}/usepde_{args.pde_guide}/'
    os.makedirs(dir_output, exist_ok=True)

    # Define shot and snapshot indices for test data.
    shot_id = 2
    snap_id = 3

    # Loop over test data samples.
    for id in range(data_num):
        print(f'Sampling for test data {id+1}.mat')
        # Load test data from .mat file.
        dict = sio.loadmat(f'../dataset/test/data{id+1}_shot{shot_id}_snap{snap_id}.mat')
        vx = dict['vx']
        vx_p = dict['vx_p']
        vz = dict['vz']
        vz_p = dict['vz_p']
        mid_vxp = dict['mid_vxp']
        mid_vzp = dict['mid_vzp']
        vp = dict['vp']
        vs = dict['vs']

        # Normalize velocity models.
        vp = normalizer_vp(vp)
        vs = normalizer_vs(vs)

        # Convert data to numpy float32 arrays.
        vx = np.array(vx, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        vx_p = np.array(vx_p, dtype=np.float32)
        vz_p = np.array(vz_p, dtype=np.float32)
        mid_vxp = np.array(mid_vxp, dtype=np.float32)
        mid_vzp = np.array(mid_vzp, dtype=np.float32)
        vp = np.array(vp, dtype=np.float32)
        vs = np.array(vs, dtype=np.float32)

        # Stack channels: combine vx and vz into a single tensor.
        vxz = np.stack((vx, vz), axis=0)
        vxz_p = np.stack((vx_p, vz_p), axis=0)
        mid_p = np.stack((mid_vxp, mid_vzp), axis=0)
        vel = np.stack((vp, vs), axis=0)

        # Convert numpy arrays to PyTorch tensors and add a batch dimension.
        vxz = th.tensor(vxz, dtype=th.float32).unsqueeze(0).to(device=device)
        vxz_p = th.tensor(vxz_p, dtype=th.float32).unsqueeze(0).to(device=device)
        mid_p = th.tensor(mid_p, dtype=th.float32).unsqueeze(0).to(device=device)
        vel = th.tensor(vel, dtype=th.float32).unsqueeze(0).to(device=device)

        b, _, w, h = vxz.shape

        # Repeat velocity tensor along the batch dimension.
        vel = vel.repeat(b, 1, 1, 1)

        model_kwargs = {}

        # Select sampling function: use DDPM or DDIM based on command-line argument.
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        loc = []  # Optional conditioning location; empty in this case.
        # If 'with_ini' flag is True, provide initial noise to the sampler.
        if args.with_ini:
            sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = sample_fn(
                model, vxz, mid_p, vel, loc, args.dh,
                (b, args.out_channels, w, h),
                scale_factor=args.scale_factor,
                noise=vxz,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        else:
            sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = sample_fn(
                model, vxz, mid_p, vel, loc, args.dh,
                (b, args.out_channels, w, h),
                scale_factor=args.scale_factor,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        # Stack intermediate samples and x_0 predictions.
        sample_all = th.stack(sample_all)
        pred_xstart = th.stack(pred_xstart)

        # Compute MSE loss between predicted and true velocity components.
        accs_vxp = criterion(sample[:, 0], vxz_p[:, 0])
        accs_vzp = criterion(sample[:, 1], vxz_p[:, 1])

        # Save the output results in a .mat file.
        sio.savemat(f'{dir_output}data{id+1}_shot{shot_id}_snap{snap_id}_out.mat', 
                {'vxp_pred': sample[:, 0].squeeze().cpu().numpy(), 
                 'vzp_pred': sample[:, 1].squeeze().cpu().numpy(),
                 'sample_all': sample_all.squeeze().cpu().numpy(),
                 'pred_xstart': pred_xstart.squeeze().cpu().numpy(),
                 'accs_vxp': accs_vxp.item(), 
                 'accs_vzp': accs_vzp.item(), 
                 'pde_loss_before': np.array(pde_loss_before, dtype=np.float32),
                 'pde_loss_after': np.array(pde_loss_after, dtype=np.float32)})

    logger.log("sampling complete")


def create_argparser():
    # Define default parameters for sampling.
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        with_ini=False,
        dh=10,
        scale_factor=200,
        model_path="./checkpoints/ema_0.999_",
    )
    # Update defaults with model and diffusion specific defaults.
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # Automatically add dictionary entries as command-line arguments.
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
