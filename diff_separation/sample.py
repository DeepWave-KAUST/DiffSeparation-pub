"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description
-----------
This script performs inference/sampling using a trained physics-informed
conditional diffusion model (PICDM) for elastic P-wave mode separation.

Given a test elastic wavefield and velocity models,

    c = (Vx, Vz, vp, vs),

the trained diffusion model generates the clean P-wave mode

    x_0 = (Vx^p, Vz^p).

The generated P-wave mode can then be compared with the reference P-wave mode
obtained by the conventional numerical separation method.

Main functions of this script
-----------------------------
1. Load a trained diffusion model checkpoint.
2. Load test wavefield samples from .npz files.
3. Normalize velocity models vp and vs.
4. Run either DDPM or DDIM sampling.
5. Optionally apply physics-guided correction during sampling.
6. Save predicted P-wave modes, intermediate samples, and evaluation metrics.

Important sampling setting in the paper
---------------------------------------
During training, the model is trained with the full 1000-step diffusion process.

During inference in the paper, DDIM accelerated sampling is used:

    use_ddim = True
    timestep_respacing = "ddim50"

This means that only 50 timesteps are selected from the original 1000-step
diffusion process, and the model performs 50-step DDIM sampling.
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
    """
    Main inference function for PICDM.

    This function loads a trained model and applies it to several test velocity
    models. For each test case, it generates the separated P-wave mode using
    either DDPM or DDIM sampling.

    In the paper's default inference setting, DDIM sampling with 50 steps is used
    to accelerate the reverse diffusion process.
    """

    # Parse command-line arguments.
    args = create_argparser().parse_args()

    # Set the device to GPU.
    device = th.device("cuda")

    # Configure logger.
    logger.configure()

    logger.log("creating model and diffusion...")

    # Create the conditional U-Net and diffusion process.
    #
    # The model architecture must exactly match the training configuration,
    # otherwise loading the pretrained checkpoint will fail.
    #
    # For PICDM, the default model input has six channels:
    #   1-2: noisy P-wave mode x_t,
    #   3-4: original elastic wavefield (Vx, Vz),
    #   5-6: velocity models (vp, vs).
    #
    # The output has two channels:
    #   output[0] = predicted Vx^p,
    #   output[1] = predicted Vz^p.
    #
    # For sampling with the paper setting, use:
    #   --use_ddim True
    #   --timestep_respacing ddim50
    #
    # This creates a SpacedDiffusion object with 50 effective reverse steps.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load the trained model checkpoint.
    # Here args.model_path is expected to be something like:
    model.load_state_dict(
        th.load(f"{args.model_path}", map_location=device)
    )

    # Move model to GPU and switch to evaluation mode.
    model.to(device=device)
    model.eval()

    # MSE criterion used to evaluate the prediction against the reference P-wave.
    criterion = th.nn.MSELoss()

    logger.log("sampling...")

    # Number of test samples. This variable is currently not used below because
    # the test cases are controlled by md_list.
    data_num = 8

    # Create output directory according to:
    #   1. DDPM or DDIM sampling,
    #   2. whether physics-guided correction is used,
    #   3. the physics guidance scale factor.
    #
    # If args.use_ddim=True and args.timestep_respacing="ddim50",
    # outputs will be saved under:
    #   ./output/ddim50/...
    if not args.use_ddim:
        if args.pde_guide:
            dir_output = (
                f"./output/ddpm/usepde_{args.pde_guide}_scale{args.scale_factor}/"
            )
        else:
            dir_output = f"./output/ddpm/usepde_{args.pde_guide}/"
    else:
        if args.pde_guide:
            dir_output = (
                f"./output/{args.timestep_respacing}/"
                f"usepde_{args.pde_guide}_scale{args.scale_factor}/"
            )
        else:
            dir_output = (
                f"./output/{args.timestep_respacing}/usepde_{args.pde_guide}/"
            )

    os.makedirs(dir_output, exist_ok=True)

    # Test shot and snapshot indices.
    #
    # These should match the file naming convention in ../dataset/test/.
    # Example:
    #   seamarid_shot1_snap5.npz
    shot_id = 1
    snap_id = 5

    # Test models used for evaluating generalization.
    #
    # These correspond to several in-distribution and out-of-distribution cases
    # discussed in the paper, such as SEAM Arid, Overthrust, Otway, and Marmousi.
    md_list = ["seamarid", "overthrust", "otway", "marmousi_small", "marmousi"]

    # Loop over all test models.
    for md in md_list:
        print(f"Sampling for {md}")

        # Load test data from .npz file.
        #
        # Expected variables:
        #   vx, vz:
        #       original coupled elastic velocity wavefield components.
        #
        #   vx_p, vz_p:
        #       reference P-wave components obtained using the conventional
        #       numerical wave-mode separation method.
        #
        #   mid_vxp, mid_vzp:
        #       precomputed right-hand-side terms for the physics-informed
        #       Laplacian constraint.
        #
        #   vp, vs:
        #       P- and S-wave velocity models.
        data = np.load(f"../dataset/test/{md}_shot{shot_id}_snap{snap_id}.npz")

        vx = data["vx"]
        vx_p = data["vx_p"]

        vz = data["vz"]
        vz_p = data["vz_p"]

        mid_vxp = data["mid_vxp"]
        mid_vzp = data["mid_vzp"]

        vp = data["vp"]
        vs = data["vs"]

        # Normalize velocity models to the same range used during training.
        #
        # The normalized vp and vs are used as conditional inputs to the U-Net.
        # This matches the paper's conditional input:
        #
        #   c = (Vx, Vz, vp, vs).
        vp = normalizer_vp(vp)
        vs = normalizer_vs(vs)

        # Convert all arrays to float32 for PyTorch inference.
        vx = np.array(vx, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)

        vx_p = np.array(vx_p, dtype=np.float32)
        vz_p = np.array(vz_p, dtype=np.float32)

        mid_vxp = np.array(mid_vxp, dtype=np.float32)
        mid_vzp = np.array(mid_vzp, dtype=np.float32)

        vp = np.array(vp, dtype=np.float32)
        vs = np.array(vs, dtype=np.float32)

        # Stack original elastic wavefield components:
        #   vxz[0] = Vx
        #   vxz[1] = Vz
        #
        # This is part of the physical condition.
        vxz = np.stack((vx, vz), axis=0)

        # Stack reference clean P-wave components:
        #   vxz_p[0] = Vx^p
        #   vxz_p[1] = Vz^p
        #
        # This is used only for evaluation, not as input to the sampler.
        vxz_p = np.stack((vx_p, vz_p), axis=0)

        # Stack precomputed physics terms for the Laplacian constraint.
        #
        # In the physics loss, the model compares:
        #   Delta predicted Vx^p with mid_p[0],
        #   Delta predicted Vz^p with mid_p[1].
        mid_p = np.stack((mid_vxp, mid_vzp), axis=0)

        # Stack normalized velocity models:
        #   vel[0] = normalized vp
        #   vel[1] = normalized vs
        vel = np.stack((vp, vs), axis=0)

        # Convert NumPy arrays to PyTorch tensors and add batch dimension.
        vxz = th.tensor(vxz, dtype=th.float32).unsqueeze(0).to(device=device)
        vxz_p = th.tensor(vxz_p, dtype=th.float32).unsqueeze(0).to(device=device)
        mid_p = th.tensor(mid_p, dtype=th.float32).unsqueeze(0).to(device=device)
        vel = th.tensor(vel, dtype=th.float32).unsqueeze(0).to(device=device)

        # Tensor shape:
        #   vxz.shape = [B, 2, H, W]
        #
        # Here b is normally 1 for single-snapshot inference.
        b, _, w, h = vxz.shape

        # Repeat velocity tensor along the batch dimension.
        #
        # Since b is normally 1 here, this does not change the tensor. It is
        # retained for compatibility with possible batched sampling.
        vel = vel.repeat(b, 1, 1, 1)

        # Extra keyword conditions. Empty here because PICDM uses direct
        # physical conditioning through vxz and vel.
        model_kwargs = {}

        # Select the sampling routine.
        #
        # If args.use_ddim=False:
        #   use standard stochastic DDPM reverse sampling.
        #
        # If args.use_ddim=True:
        #   use DDIM sampling. With args.timestep_respacing="ddim50", this
        #   performs 50-step accelerated sampling, as used in the paper.
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Optional source/time metadata.
        #
        # In the current UNetModel.forward(), loc is included in the interface
        # but is not explicitly used. Therefore, an empty list is passed here.
        loc = []

        # Run the reverse diffusion sampler.
        #
        # The sampler starts from Gaussian noise by default and generates the
        # clean P-wave mode sample.
        #
        # If args.with_ini=True, the sampler uses vxz as the initial noise/input.
        # This is an experimental option and is not the standard unconditional
        # Gaussian initialization used in diffusion sampling.
        if args.with_ini:
            sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = (
                sample_fn(
                    model,
                    vxz,
                    mid_p,
                    vel,
                    loc,
                    args.dh,
                    (b, args.out_channels, w, h),
                    scale_factor=args.scale_factor,
                    noise=vxz,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            )
        else:
            sample, sample_all, pred_xstart, pde_loss_before, pde_loss_after = (
                sample_fn(
                    model,
                    vxz,
                    mid_p,
                    vel,
                    loc,
                    args.dh,
                    (b, args.out_channels, w, h),
                    scale_factor=args.scale_factor,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            )

        # Stack intermediate samples and predicted x_0 values collected during
        # the sampling trajectory.
        #
        # sample_all:
        #   intermediate generated samples at selected reverse steps.
        #
        # pred_xstart:
        #   network predictions of x_0 at selected reverse steps.
        sample_all = th.stack(sample_all)
        pred_xstart = th.stack(pred_xstart)

        # Compute MSE between the generated P-wave mode and the reference
        # numerical P-wave mode.
        #
        # These correspond to the quantitative evaluation used to compare PICDM
        # with the conventional separation result.
        accs_vxp = criterion(sample[:, 0], vxz_p[:, 0])
        accs_vzp = criterion(sample[:, 1], vxz_p[:, 1])

        # Save output results to a MATLAB .mat file for visualization and
        # quantitative analysis.
        #
        # Saved variables:
        #   vxp_pred, vzp_pred:
        #       final generated P-wave components.
        #
        #   sample_all:
        #       intermediate reverse-diffusion samples.
        #
        #   pred_xstart:
        #       predicted clean P-wave modes along the sampling process.
        #
        #   accs_vxp, accs_vzp:
        #       MSE errors relative to the numerical reference.
        #
        #   pde_loss_before, pde_loss_after:
        #       physics residuals before and after physics-guided correction.
        sio.savemat(
            f"{dir_output}{md}_shot{shot_id}_snap{snap_id}_out.mat",
            {
                "vxp_pred": sample[:, 0].squeeze().cpu().numpy(),
                "vzp_pred": sample[:, 1].squeeze().cpu().numpy(),
                "sample_all": sample_all.squeeze().cpu().numpy(),
                "pred_xstart": pred_xstart.squeeze().cpu().numpy(),
                "accs_vxp": accs_vxp.item(),
                "accs_vzp": accs_vzp.item(),
                "pde_loss_before": np.array(pde_loss_before, dtype=np.float32),
                "pde_loss_after": np.array(pde_loss_after, dtype=np.float32),
            },
        )

    logger.log("sampling complete")


def create_argparser():
    """
    Create the command-line argument parser for PICDM sampling.

    The sampling defaults are combined with the model/diffusion defaults defined
    in script_util.py.

    Important arguments
    -------------------
    clip_denoised : bool
        If True, clip predicted x_0 to the range [-1, 1].

    use_ddim : bool
        If True, use DDIM sampling instead of standard DDPM sampling.

    timestep_respacing : str
        This argument comes from model_and_diffusion_defaults().

        For paper-style accelerated inference, set:

            --timestep_respacing ddim50

        This selects 50 timesteps from the original 1000-step diffusion schedule.

    pde_guide : bool
        This argument also comes from model_and_diffusion_defaults().

        If True, apply physics-guided correction at each reverse sampling step:

            x <- x - eta * grad L_phys.

    scale_factor : float
        Controls the strength of the physics-guided correction. The paper uses
        a scale factor of 5.

    dh : float
        Spatial grid spacing used by the finite-difference Laplacian operator.
        The paper uses dh = 10 m.

    model_path : str
        Prefix of the pretrained model checkpoint path.
    """

    # Sampling-specific defaults.
    defaults = dict(
        # Clip predicted x_0 into [-1, 1].
        clip_denoised=True,

        # Use DDIM sampling by default.
        # To reproduce the paper's accelerated inference setting, also set:
        #   --timestep_respacing ddim50
        use_ddim=True,

        # If True, use vxz as initial noise/input. The standard diffusion
        # sampling mode keeps this False and starts from Gaussian noise.
        with_ini=False,

        # Spatial grid interval used in physics loss/guidance.
        dh=10,

        # Physics-guidance scale factor.
        scale_factor=5,

        # Checkpoint filename prefix.
        model_path="../trained_model/trained_model_singlefreq.pt",
    )

    # Add model and diffusion defaults:
    #   in_channels=6,
    #   out_channels=2,
    #   diffusion_steps=1000,
    #   noise_schedule="cosine",
    #   timestep_respacing="",
    #   predict_xstart=True,
    #   pde_guide=False,
    #   fd_order=8,
    # etc.
    #
    # NOTE:
    # model_and_diffusion_defaults() sets timestep_respacing="" by default.
    # For paper-style sampling, you should override it from the command line:
    #
    #   --timestep_respacing ddim50
    #
    # or change the sampling default explicitly after this update.
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()

    # Automatically add all defaults as command-line arguments.
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
