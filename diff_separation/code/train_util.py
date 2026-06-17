import copy
import functools
import os
import blobfile as bf
import numpy as np
import torch as th
from torch.optim import AdamW
from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


# Initial logarithmic loss scale used when mixed-precision training is enabled.
# This follows the original improved-diffusion implementation. In the current
# PICDM training, use_fp16 can be disabled for standard full-precision training.
INITIAL_LOG_LOSS_SCALE = 20.0


# Directory used to save diffusion-model checkpoints.
# In this implementation, only EMA checkpoints are saved by default.
dir_checkpoints = "./checkpoints/"
os.makedirs(dir_checkpoints, exist_ok=True)


class TrainLoop:
    """
    Training loop for the physics-informed conditional diffusion model (PICDM).

    This class controls the complete optimization process of the denoising
    diffusion network used for elastic P-wave mode separation.

    In the paper, the PICDM learns to generate the clean P-wave mode

        x_0 = (Vx^p, Vz^p)

    from noisy P-wave samples x_t under physical conditions

        c = (Vx, Vz, vp, vs),

    where (Vx, Vz) are the original coupled elastic velocity wavefields and
    (vp, vs) are the subsurface P- and S-wave velocity models.

    At each training iteration, this loop:

        1. Loads a mini-batch from the dataset:
               batch_vxz_p : clean P-wave target x_0 = (Vx^p, Vz^p)
               batch_vxz   : original elastic wavefield (Vx, Vz)
               batch_mid_p : auxiliary/intermediate P-wave information
               batch_vel   : velocity condition (vp, vs)
               batch_loc   : source/time metadata
               cond        : optional keyword-condition dictionary

        2. Randomly samples diffusion timesteps t.

        3. Calls diffusion.training_losses(...), where the forward diffusion
           process corrupts x_0 into x_t and the model predicts the clean
           P-wave mode.

        4. Backpropagates the total PICDM training loss, which corresponds to

               L_train = L_data + lambda * L_phys,

           where L_data enforces agreement with the reference P-wave label and
           L_phys enforces the Laplacian P-wave separation equation.

        5. Updates the model parameters and their exponential moving average
           (EMA), which is commonly used for stable diffusion-model sampling.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=1e-4,
        lr_anneal_steps=0,
    ):
        """
        Initialize the PICDM training loop.

        Parameters
        ----------
        model : torch.nn.Module
            The denoising network f_theta. In the paper, this is a U-Net-based
            diffusion model that predicts the clean P-wave mode.

        diffusion : object
            Diffusion process object. It provides training_losses(...), which
            implements the forward diffusion process, denoising objective, and
            physics-informed loss terms.

        data : iterator
            Infinite data iterator returned by load_data(...). Each iteration
            provides one batch of elastic wavefield separation samples.

        batch_size : int
            Number of wavefield snapshots per mini-batch.

        lr : float
            Learning rate for AdamW.

        ema_rate : float or str
            Exponential moving average rate. EMA parameters are typically used
            during inference because they provide more stable generated results.

        log_interval : int
            Number of optimization steps between logging events.

        save_interval : int
            Number of optimization steps between checkpoint saves.

        resume_checkpoint : str
            Path to a checkpoint for resuming training. If empty or None, the
            model starts from the current initialized parameters.

        use_fp16 : bool
            If True, use mixed-precision training. Otherwise, use standard
            full-precision training.

        fp16_scale_growth : float
            Growth factor for the dynamic loss scale in fp16 training.

        schedule_sampler : object, optional
            Sampler for diffusion timesteps. If None, a UniformSampler is used.
            This means each diffusion timestep is sampled uniformly during
            training.

        weight_decay : float
            Weight decay used in AdamW.

        lr_anneal_steps : int
            Number of steps over which to anneal the learning rate. If zero,
            no learning-rate annealing is applied.
        """
        self.model = model

        # Use the same device as the model parameters.
        self.device = next(model.parameters()).device

        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr

        # Allow one or multiple EMA rates.
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint

        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        # The timestep sampler determines which diffusion step t is used for
        # each training sample. In the PICDM objective, the clean P-wave target
        # x_0 is corrupted to x_t according to the sampled timestep.
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # Spatial grid interval used in the physics-informed finite-difference
        # operators. In the paper, the wavefield grid spacing is 10 m.
        # This value is passed to training_losses so that the physics loss can
        # compute spatial derivatives such as d_xx, d_zz, d_xz, and Laplacian.
        self.dh = 10

        self.step = 0
        self.resume_step = 0

        # Effective global batch size. This code does not explicitly use
        # distributed aggregation, so it equals the local batch size here.
        self.global_batch = self.batch_size

        # Model parameters and master parameters. For fp32 training, these are
        # identical. For fp16 training, master_params store fp32 copies.
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Resume model parameters if a checkpoint is provided.
        self._load_and_sync_parameters()

        # Setup mixed-precision parameters if needed.
        if self.use_fp16:
            self._setup_fp16()

        # AdamW optimizer used for training the denoising network.
        # The paper uses AdamW with a fixed learning rate.
        self.opt = th.optim.AdamW(
            self.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Load optimizer and EMA states when resuming; otherwise initialize EMA
        # parameters from the current model parameters.
        if self.resume_step:
            self._load_optimizer_state()

            # Model was resumed, either due to a restart or a checkpoint being
            # specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        """
        Load model parameters from a checkpoint if available.

        The checkpoint stores the denoising network f_theta. Resuming from a
        checkpoint allows training to continue from a previous PICDM run.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(resume_checkpoint, map_location=self.device)
            )

    def _load_ema_parameters(self, rate):
        """
        Load EMA parameters corresponding to a resumed checkpoint.

        EMA parameters are important for diffusion models because sampling from
        the EMA network often produces more stable and accurate separated P-wave
        modes than sampling from the raw training weights.
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

            state_dict = th.load(
                ema_checkpoint, map_location=self.device
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """
        Load the optimizer state when resuming training.

        This restores AdamW moments and allows the training dynamics to continue
        smoothly from the saved checkpoint.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")

            # NOTE:
            # As above, PyTorch typically uses th.load(...) here.
            state_dict = th.load(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Prepare mixed-precision training.

        The model is converted to fp16, while fp32 master parameters are kept
        for stable optimization.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        Run the full PICDM training process.

        The loop repeatedly fetches batches from the seismic wavefield dataset,
        performs one optimization step, logs losses, and saves EMA checkpoints.

        The loop stops only when lr_anneal_steps is reached. If lr_anneal_steps
        is zero, this loop runs indefinitely unless externally interrupted.
        """
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            # Load one mini-batch:
            #
            # batch_vxz_p:
            #     Clean P-wave target x_0 = (Vx^p, Vz^p).
            #
            # batch_vxz:
            #     Original coupled elastic wavefield (Vx, Vz), used as a
            #     physical condition and in the physics-informed residual.
            #
            # batch_mid_p:
            #     Auxiliary/intermediate P-wave field, passed to the diffusion
            #     loss function for possible model-specific conditioning.
            #
            # batch_vel:
            #     Normalized velocity models (vp, vs), used as conditional input.
            #
            # batch_loc:
            #     Source location and snapshot-time metadata.
            #
            # cond:
            #     Optional dictionary for compatibility with improved-diffusion.
            batch_vxz_p, batch_vxz, batch_mid_p, batch_vel, batch_loc, cond = next(
                self.data
            )

            self.run_step(
                batch_vxz_p,
                batch_vxz,
                batch_mid_p,
                batch_vel,
                batch_loc,
                cond,
            )

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()

                # Used only for integration tests to stop training early.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1

        # Save the last checkpoint if it was not already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_vxz_p, batch_vxz, batch_mid_p, batch_vel, batch_loc, cond):
        """
        Perform one PICDM optimization step.

        This includes:
            1. Forward diffusion and loss computation.
            2. Backpropagation.
            3. Optimizer update.
            4. EMA update.
            5. Logging.
        """
        self.forward_backward(
            batch_vxz_p,
            batch_vxz,
            batch_mid_p,
            batch_vel,
            batch_loc,
            cond,
        )

        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()

        self.log_step()

    def forward_backward(self, batch_vxz_p, batch_vxz, batch_mid_p, batch_vel, batch_loc, cond):
        """
        Compute the PICDM training loss and backpropagate gradients.

        This function is the core training step connecting the dataset to the
        diffusion objective.

        In the paper's notation:

            batch_vxz_p = x_0 = (Vx^p, Vz^p)

        The diffusion object internally samples or constructs x_t from x_0:

            x_t = sqrt(alpha_bar_t) * x_0
                  + sqrt(1 - alpha_bar_t) * epsilon.

        Then the model predicts the clean P-wave mode from x_t and the physical
        conditions. The returned loss dictionary usually contains terms such as:

            loss:
                Total training loss, typically L_data + lambda * L_phys.

            data loss:
                Difference between predicted and reference P-wave modes.

            physics loss:
                Residual of the Laplacian P-wave separation equation.

        The exact keys depend on the implementation of diffusion.training_losses.
        """
        # Clear gradients from the previous iteration.
        zero_grad(self.model_params)

        # Move all wavefield and condition tensors to the model device.
        batch_vxz_p = batch_vxz_p.to(self.device)
        batch_vxz = batch_vxz.to(self.device)
        batch_mid_p = batch_mid_p.to(self.device)
        batch_vel = batch_vel.to(self.device)
        batch_loc = batch_loc.to(self.device)

        # Sample diffusion timestep t for each sample in the batch.
        # weights are importance weights associated with the timestep sampler.
        t, weights = self.schedule_sampler.sample(batch_vxz_p.shape[0], self.device)

        # Prepare a partial function for computing the diffusion training losses.
        #
        # The arguments passed here contain all quantities required by PICDM:
        #
        #   self.model:
        #       Denoising network f_theta.
        #
        #   batch_vxz_p:
        #       Clean P-wave target x_0.
        #
        #   batch_vxz:
        #       Original elastic wavefield (Vx, Vz), used as condition and for
        #       the physics residual in Zhu's separation equation.
        #
        #   batch_mid_p:
        #       Auxiliary/intermediate P-wave information.
        #
        #   batch_vel:
        #       Velocity condition (vp, vs).
        #
        #   batch_loc:
        #       Source and time metadata.
        #
        #   self.dh:
        #       Spatial grid interval, used by finite-difference derivatives in
        #       the physics-informed loss.
        #
        #   t:
        #       Randomly sampled diffusion timestep.
        #
        #   model_kwargs:
        #       Optional extra conditions.
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch_vxz_p,
            batch_vxz,
            batch_mid_p,
            batch_vel,
            batch_loc,
            self.dh,
            t,
            model_kwargs=cond,
        )

        # Compute the full loss dictionary.
        # This is where the paper's data loss and physics-informed loss are
        # expected to be evaluated.
        losses = compute_losses()

        # If a loss-aware timestep sampler is used, update the sampler so that
        # more difficult diffusion timesteps can be sampled more frequently.
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        # Apply timestep importance weights and average over the batch.
        loss = (losses["loss"] * weights).mean()

        # Log all available loss terms, including quartile-wise statistics over
        # diffusion timesteps. This helps diagnose whether early or late diffusion
        # steps are harder for P-wave reconstruction.
        log_loss_dict(
            self.diffusion,
            t,
            {k: v * weights for k, v in losses.items()},
        )

        # Backpropagate the total PICDM loss.
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize_fp16(self):
        """
        Optimizer step for mixed-precision training.

        This function handles loss scaling, checks for numerical overflow,
        updates AdamW parameters, updates EMA parameters, and copies fp32
        master parameters back to the fp16 model.
        """
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)

        # Undo loss scaling before the optimizer step.
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))

        self._log_grad_norm()
        self._anneal_lr()

        self.opt.step()

        # Update EMA copy of model parameters for stable diffusion sampling.
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

        master_params_to_model_params(self.model_params, self.master_params)

        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Optimizer step for standard full-precision training.

        The denoising model is updated using AdamW, and EMA parameters are
        updated after each optimization step.
        """
        self._log_grad_norm()
        self._anneal_lr()

        self.opt.step()

        # EMA parameters are saved by default and are typically preferred for
        # inference/sampling in diffusion models.
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """
        Log the global gradient norm.

        This is useful for monitoring training stability, especially because the
        PICDM loss contains finite-difference physics residuals that may amplify
        gradients if the physical loss weight is too large.
        """
        sqsum = 0.0

        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()

        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Optionally anneal the learning rate.

        If lr_anneal_steps is zero, the learning rate remains fixed. This matches
        the fixed-learning-rate training setting described in the paper.
        """
        if not self.lr_anneal_steps:
            return

        frac_done = 0.8 * (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)

        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        Log the current training step and number of processed samples.
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv(
            "samples",
            (self.step + self.resume_step + 1) * self.global_batch,
        )

        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save EMA checkpoints of the PICDM denoising model.

        In the current implementation, the raw model checkpoint and optimizer
        checkpoint are commented out, while EMA checkpoints are saved. This is
        reasonable for diffusion-model inference, where EMA weights are commonly
        used for more stable P-wave generation.
        """

        def save_checkpoint(rate, params):
            """
            Save a single checkpoint corresponding to one EMA rate.

            Parameters
            ----------
            rate : float
                EMA decay rate. If rate is zero, the raw model parameters would
                be saved. Otherwise, EMA parameters are saved.

            params : list
                Parameters to be converted into a model state dict and saved.
            """
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")

            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"

            with bf.BlobFile(bf.join(dir_checkpoints, filename), "wb") as f:
                th.save(state_dict, f)

        # Raw model checkpoint saving is disabled by default.
        # save_checkpoint(0, self.master_params)

        # Save EMA checkpoints.
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Optimizer checkpoint saving is disabled by default.
        # Enable this block if exact training resumption is required.
        #
        # with bf.BlobFile(
        #     bf.join(dir_checkpoints, f"opt{(self.step + self.resume_step):06d}.pt"),
        #     "wb",
        # ) as f:
        #     th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master parameters into a PyTorch model state dict.

        For fp16 training, master parameters are first converted back to the
        original model parameter structure.
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(),
                master_params,
            )

        state_dict = self.model.state_dict()

        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]

        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Convert a model state dict into master parameters.

        This is used when loading EMA parameters from a checkpoint.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]

        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse the training step from a checkpoint filename.

    Expected filename format:

        path/to/modelNNNNNN.pt

    where NNNNNN is the number of training steps.

    Parameters
    ----------
    filename : str
        Checkpoint filename.

    Returns
    -------
    int
        Parsed step number. Returns 0 if parsing fails.
    """
    split = filename.split("model")

    if len(split) < 2:
        return 0

    split1 = split[-1].split(".")[0]

    try:
        return int(split1)
    except ValueError:
        return 0


def parse_dataname_from_filename(filename):
    """
    Parse a data name or suffix from a filename containing the string 'gaussian5'.

    This helper is not used in the main training loop. It appears to be a
    project-specific utility for extracting a dataset identifier.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    str or int
        Parsed suffix after 'gaussian5', or 0 if parsing fails.
    """
    split = filename.split("gaussian5")

    if len(split) < 2:
        return 0

    split1 = split[-1].split(".")[0]

    try:
        return split1
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Return the logging directory.

    If DIFFUSION_BLOB_LOGDIR is defined, use it. Otherwise, use the default
    logger directory.
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    Automatically find a checkpoint for resuming training.

    This function currently returns None. It can be customized for a specific
    computing environment, for example to automatically find the latest
    checkpoint on shared storage or cloud storage.
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Find the EMA checkpoint associated with a main checkpoint.

    Parameters
    ----------
    main_checkpoint : str
        Path to the main model checkpoint.

    step : int
        Training step number.

    rate : float
        EMA rate.

    Returns
    -------
    str or None
        Path to the EMA checkpoint if it exists; otherwise None.
    """
    if main_checkpoint is None:
        return None

    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)

    if bf.exists(path):
        return path

    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log diffusion losses and their timestep-dependent statistics.

    Diffusion models may behave differently at different noise levels. Therefore,
    besides logging the mean value of each loss term, this function also logs
    quartile statistics over diffusion timesteps.

    For PICDM, this can help reveal whether the data loss or physics-informed
    loss is more difficult at early, middle, or late denoising stages.

    Parameters
    ----------
    diffusion : object
        Diffusion object containing num_timesteps.

    ts : torch.Tensor
        Sampled diffusion timesteps for the current mini-batch.

    losses : dict
        Dictionary of loss terms returned by diffusion.training_losses(...).
    """
    for key, values in losses.items():
        # Log the average value of each loss term.
        logger.logkv_mean(key, values.mean().item())

        # Log loss statistics grouped by timestep quartile.
        # q0 corresponds to low-noise timesteps, while q3 corresponds to
        # high-noise timesteps, depending on the diffusion schedule.
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
