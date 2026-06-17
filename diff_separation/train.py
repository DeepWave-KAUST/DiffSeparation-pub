"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com
Description: This script is the main training program for a physics-informed conditional diffusion model.
It configures logging, creates the model and diffusion process, loads the training dataset, and
initiates the training loop using PyTorch. The script also supports command-line arguments for
flexible configuration of the training process.
"""

import argparse

# Import custom modules for logging, dataset loading, model creation, diffusion process,
# and training utilities.
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
    # Parse command-line arguments using the auto-generated argument parser.
    args = create_argparser().parse_args()

    # Configure the logger for outputting training logs.
    logger.configure()

    # Set the device to GPU (CUDA) for training.
    device = th.device('cuda')

    logger.log("creating model and diffusion...")

    # Create the model and the diffusion process using the provided arguments.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Transfer the model to the GPU.
    model.to(device)

    # Create the schedule sampler, which can sample uniformly or according to the loss values.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # Load the training dataset from the specified directory with the given batch size.
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    # Initialize and run the training loop with the configured parameters.
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
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
    Automatically creates a command-line argument parser from a default parameters dictionary.
    """
    defaults = dict(
        data_dir="../dataset/train/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=5e-6,
        lr_anneal_steps=600000,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.999",  # Comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    # Update the defaults with model and diffusion specific default parameters.
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # Automatically add dictionary keys as command-line arguments.
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
