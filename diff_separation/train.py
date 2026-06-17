"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description
-----------
Main training script for the physics-informed conditional diffusion model (PICDM).

This script configures and launches the training process for elastic P-wave mode
separation. In the PICDM framework, the model learns to recover the clean P-wave
mode

    x_0 = (Vx^p, Vz^p)

from a noisy P-wave mode x_t, conditioned on the original coupled elastic
wavefield and velocity models:

    c = (Vx, Vz, vp, vs).

The training objective combines the diffusion reconstruction loss and, when
enabled, the physics-informed Laplacian loss derived from elastic wave-mode
separation equations:

    L_train = L_data + lambda * L_phys.

This script performs the following steps:

    1. Parse command-line arguments.
    2. Configure the logger.
    3. Create the conditional U-Net denoising model.
    4. Create the Gaussian/Spaced diffusion process.
    5. Load the elastic wavefield training dataset.
    6. Create the timestep schedule sampler.
    7. Launch the training loop.
"""

import argparse

# Import project-specific modules.
# logger:
#     Handles logging of training loss, gradient norm, step number, etc.
#
# load_data:
#     Loads training samples containing:
#         - clean P-wave target (Vx^p, Vz^p),
#         - original elastic wavefield (Vx, Vz),
#         - velocity models (vp, vs),
#         - auxiliary physics terms and metadata.
#
# create_named_schedule_sampler:
#     Creates a diffusion timestep sampler, such as uniform timestep sampling.
#
# script_util:
#     Provides default model/diffusion settings and factory functions.
#
# TrainLoop:
#     Implements the optimization loop for PICDM.
from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop

import torch as th


def main():
    """
    Main entry point for PICDM training.

    This function creates all necessary components and starts the training loop.
    The default configuration follows the paper's main training setting:

        - 1000 diffusion timesteps,
        - cosine noise schedule,
        - x_0 prediction, where x_0 = (Vx^p, Vz^p),
        - six-channel conditional U-Net input,
        - two-channel P-wave output,
        - physics-informed loss enabled,
        - AdamW optimizer with learning rate 1e-4,
        - batch size 16,
        - EMA rate 0.999.

    During training, timestep_respacing is set to "" by default. This means the
    full 1000-step diffusion schedule is used. For accelerated inference, the
    sampling script can instead set timestep_respacing="ddim50" to perform
    50-step DDIM sampling.
    """
    # Parse all command-line arguments.
    # The parser is automatically generated from default training parameters
    # and model/diffusion parameters.
    args = create_argparser().parse_args()

    # Configure logging directory and output behavior.
    logger.configure()

    # Use CUDA for training.
    # If CPU fallback is desired, replace this with:
    #     device = th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device("cuda")

    logger.log("creating model and diffusion...")

    # Create the conditional U-Net model and the diffusion object.
    #
    # model:
    #     The denoising network f_theta. It receives:
    #         x_t  : noisy P-wave mode,
    #         vxz  : original elastic wavefield (Vx, Vz),
    #         vel  : velocity models (vp, vs),
    #         t    : diffusion timestep,
    #     and predicts the clean P-wave mode x_0 = (Vx^p, Vz^p).
    #
    # diffusion:
    #     The Gaussian/Spaced diffusion process. It handles:
    #         - forward noising q(x_t | x_0),
    #         - training loss computation,
    #         - optional physics-informed loss,
    #         - reverse sampling during inference.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Move the model to GPU.
    model.to(device)

    # Create a timestep sampler for training.
    #
    # With schedule_sampler="uniform", each diffusion timestep is sampled
    # uniformly. This means the network learns to denoise P-wave modes across
    # the complete diffusion trajectory, from low-noise to high-noise states.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # Create the infinite training-data iterator.
    #
    # Each batch returned by load_data contains:
    #     batch_vxz_p:
    #         Clean P-wave target x_0 = (Vx^p, Vz^p).
    #
    #     batch_vxz:
    #         Original coupled elastic velocity wavefield (Vx, Vz).
    #
    #     batch_mid_p:
    #         Precomputed physics-related terms used in the Laplacian
    #         physics-informed loss.
    #
    #     batch_vel:
    #         Normalized velocity models (vp, vs).
    #
    #     batch_loc:
    #         Source location and snapshot-time metadata.
    #
    #     cond:
    #         Optional condition dictionary retained for compatibility with
    #         improved-diffusion style code.
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    # Initialize and run the training loop.
    #
    # TrainLoop repeatedly:
    #     1. samples a mini-batch,
    #     2. samples diffusion timesteps t,
    #     3. corrupts the clean P-wave mode x_0 into x_t,
    #     4. predicts x_0 using the conditional U-Net,
    #     5. computes the diffusion MSE loss and physics-informed loss,
    #     6. updates model parameters with AdamW,
    #     7. updates EMA parameters,
    #     8. saves checkpoints.
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """
    Create the command-line argument parser for PICDM training.

    The parser combines:

        1. Training-loop defaults:
           data directory, learning rate, batch size, EMA rate, logging interval,
           checkpoint interval, etc.

        2. Model and diffusion defaults:
           U-Net architecture, diffusion steps, noise schedule, timestep
           respacing, physics-loss options, finite-difference order, etc.

    Important default settings
    --------------------------
    data_dir = "../dataset/train/"
        Directory containing the training .npz files.

    schedule_sampler = "uniform"
        Uniformly samples diffusion timesteps during training.

    lr = 1e-4
        AdamW learning rate.

    batch_size = 16
        Mini-batch size. This matches the paper's training setting.

    ema_rate = "0.999"
        Exponential moving average rate for model parameters. EMA weights are
        typically used for more stable diffusion sampling.

    timestep_respacing = ""
        This default is inherited from model_and_diffusion_defaults().
        During training, an empty string means the model uses the full 1000-step
        diffusion process.

    use_physicsloss = True
        Enables the physics-informed Laplacian loss during training.

    fd_order = 8
        Uses an eighth-order finite-difference Laplacian operator in the
        physics-informed loss.
    """
    defaults = dict(
        # Path to the training dataset.
        data_dir="../dataset/train/",

        # Timestep sampling strategy.
        # "uniform" is the standard setting.
        schedule_sampler="uniform",

        # Optimizer settings.
        lr=1e-4,
        weight_decay=5e-6,

        # Number of training steps for learning-rate annealing.
        # If lr_anneal_steps is nonzero, TrainLoop will gradually reduce lr.
        lr_anneal_steps=600000,

        # Mini-batch size.
        batch_size=16,

        # EMA decay rate. A comma-separated string can be used for multiple
        # EMA rates, e.g., "0.999,0.9999".
        ema_rate="0.999",

        # Logging and checkpointing intervals.
        log_interval=100,
        save_interval=10000,

        # Resume checkpoint path. Empty string means training starts from scratch.
        resume_checkpoint="",

        # Mixed-precision training options.
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    # Add model and diffusion defaults, including:
    #     in_channels=6,
    #     out_channels=2,
    #     diffusion_steps=1000,
    #     noise_schedule="cosine",
    #     timestep_respacing="",
    #     predict_xstart=True,
    #     use_physicsloss=True,
    #     pde_guide=False,
    #     fd_order=8.
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()

    # Automatically convert every entry in defaults into a command-line argument.
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
