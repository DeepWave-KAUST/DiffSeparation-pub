import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Select a reduced set of timesteps from the original diffusion process.

    In the original diffusion model, the forward and reverse processes may use
    a large number of timesteps, e.g., T = 1000. However, during inference, it is
    often unnecessary and computationally expensive to run all reverse diffusion
    steps. This function constructs a smaller subset of timesteps so that the
    model can perform accelerated sampling.

    In the PICDM paper, this idea is used together with DDIM sampling to reduce
    the reverse generation process from 1000 steps to fewer steps, such as 50
    or 10, while maintaining accurate P-wave mode separation.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps in the original diffusion process. For example,
        this is usually 1000 in standard DDPM training.

    section_counts : list[int] or str
        Number of timesteps to keep in each section of the original diffusion
        trajectory.

        Examples
        --------
        If num_timesteps = 300 and section_counts = [10, 15, 20], the original
        trajectory is divided into three sections of approximately 100 steps
        each. Then:
            - 10 timesteps are selected from the first section,
            - 15 timesteps are selected from the second section,
            - 20 timesteps are selected from the third section.

        If section_counts is a string of the form "ddimN", the function uses
        a fixed-stride timestep selection similar to DDIM, where N is the desired
        number of sampling steps. For example:
            - "ddim50" selects 50 timesteps,
            - "ddim10" selects 10 timesteps.

    Returns
    -------
    set[int]
        A set of selected timestep indices from the original diffusion process.

    Notes
    -----
    The selected timesteps are later used by SpacedDiffusion to construct a new
    diffusion process with fewer effective steps. The new process preserves the
    same cumulative alpha values at the retained timesteps, allowing accelerated
    sampling while remaining consistent with the original diffusion schedule.
    """
    if isinstance(section_counts, str):
        # DDIM-style fixed striding.
        # For example, "ddim50" means we want exactly 50 timesteps selected from
        # the original diffusion trajectory.
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])

            # Search for an integer stride that yields exactly desired_count
            # timesteps from range(0, num_timesteps, stride).
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))

            raise ValueError(
                f"cannot create exactly {desired_count} steps "
                f"from {num_timesteps} diffusion steps with an integer stride"
            )

        # If section_counts is a comma-separated string, convert it to a list.
        # Example: "10,15,20" -> [10, 15, 20].
        section_counts = [int(x) for x in section_counts.split(",")]

    # Split the original diffusion trajectory into len(section_counts) sections.
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)

    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        # Distribute any extra timesteps over the earliest sections.
        size = size_per + (1 if i < extra else 0)

        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )

        # Compute the fractional stride inside this section.
        # If only one timestep is requested, simply take the first one.
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        cur_idx = 0.0
        taken_steps = []

        # Select approximately evenly spaced timesteps from this section.
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride

        all_steps += taken_steps
        start_idx += size

    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    Diffusion process with a reduced number of effective timesteps.

    SpacedDiffusion wraps the original GaussianDiffusion process and keeps only
    a selected subset of timesteps. This allows the same trained denoising model
    to be used with fewer reverse diffusion steps during sampling.

    In the PICDM framework, this is important because elastic wave-mode
    separation should be computationally efficient. Instead of using all 1000
    reverse diffusion steps, we can use a reduced schedule such as "ddim50" or
    "ddim10" to accelerate P-wave mode generation.

    Parameters
    ----------
    use_timesteps : sequence or set
        Timesteps selected from the original diffusion process. These are
        usually produced by space_timesteps(...).

    **kwargs
        Arguments used to construct the original GaussianDiffusion object,
        including betas, model_mean_type, model_var_type, loss_type, and
        PICDM-specific options such as use_physicsloss and pde_guide.

    Attributes
    ----------
    use_timesteps : set[int]
        Selected timesteps retained from the original diffusion process.

    timestep_map : list[int]
        Mapping from the reduced diffusion timestep index to the corresponding
        original diffusion timestep index.

    original_num_steps : int
        Number of timesteps in the original diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        # Store the selected original timesteps.
        self.use_timesteps = set(use_timesteps)

        # timestep_map maps new reduced timesteps to original timesteps.
        # For example, if timestep_map[5] = 250, then reduced timestep 5
        # corresponds to original diffusion timestep 250.
        self.timestep_map = []

        # Number of timesteps in the original full diffusion process.
        self.original_num_steps = len(kwargs["betas"])

        # Build the original diffusion process first. We use its cumulative
        # alpha schedule to construct a mathematically consistent reduced
        # beta schedule.
        base_diffusion = GaussianDiffusion(**kwargs)

        last_alpha_cumprod = 1.0
        new_betas = []

        # Construct a new beta schedule that exactly matches the cumulative
        # alpha values at the selected timesteps.
        #
        # This ensures that the shortened diffusion process preserves the same
        # noise levels as the original process at the retained timesteps.
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # Replace the original beta schedule with the reduced beta schedule.
        kwargs["betas"] = np.array(new_betas)

        # Initialize the GaussianDiffusion base class using the reduced schedule.
        super().__init__(**kwargs)

    def p_mean_variance(
        self,
        model,
        *args,
        **kwargs,
    ):  # pylint: disable=signature-differs
        """
        Compute p_theta(x_{t-1} | x_t, c) using the wrapped timestep mapping.

        The reduced diffusion process uses fewer timestep indices, but the
        denoising model was trained with the original timestep indexing. The
        wrapped model converts reduced timestep indices back to their original
        timestep values before calling the network.
        """
        return super().p_mean_variance(
            self._wrap_model(model),
            *args,
            **kwargs,
        )

    def training_losses(
        self,
        model,
        *args,
        **kwargs,
    ):  # pylint: disable=signature-differs
        """
        Compute training losses using the wrapped timestep mapping.

        This makes it possible to train or evaluate with a spaced diffusion
        schedule while still passing original timestep values to the denoising
        network.
        """
        return super().training_losses(
            self._wrap_model(model),
            *args,
            **kwargs,
        )

    def _wrap_model(self, model):
        """
        Wrap the denoising model so reduced timesteps are mapped to original ones.

        If the model is already wrapped, return it directly. Otherwise, create
        a _WrappedModel that performs timestep conversion before calling the
        original neural network.
        """
        if isinstance(model, _WrappedModel):
            return model

        return _WrappedModel(
            model,
            self.timestep_map,
            self.rescale_timesteps,
            self.original_num_steps,
        )

    def _scale_timesteps(self, t):
        """
        Return timesteps unchanged.

        In SpacedDiffusion, timestep scaling is handled inside _WrappedModel,
        because the wrapper first maps reduced timesteps to original timesteps.
        """
        return t


class _WrappedModel:
    """
    Model wrapper that maps reduced diffusion timesteps to original timesteps.

    The denoising network was trained with the original diffusion schedule, for
    example with 1000 timesteps. When using SpacedDiffusion, the current timestep
    index may only range from 0 to the number of selected timesteps minus one.
    This wrapper converts the reduced timestep index back to the corresponding
    original timestep before calling the model.

    This is essential for accelerated PICDM sampling because the neural network
    still needs to receive timestep embeddings that are consistent with the
    original diffusion training schedule.
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        """
        Initialize the wrapped model.

        Parameters
        ----------
        model : torch.nn.Module
            Original PICDM denoising network.

        timestep_map : list[int]
            Mapping from reduced timestep index to original timestep index.

        rescale_timesteps : bool
            Whether to rescale timesteps to the standard 0--1000 range before
            passing them to the model.

        original_num_steps : int
            Number of timesteps in the original diffusion process.
        """
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, vxz, vel, loc, ts, **kwargs):
        """
        Call the denoising network with mapped original timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Current noisy P-wave mode x_t.

        vxz : torch.Tensor
            Original coupled elastic wavefield condition (Vx, Vz).

        vel : torch.Tensor
            Velocity-model condition (vp, vs).

        loc : torch.Tensor
            Source/time metadata.

        ts : torch.Tensor
            Reduced diffusion timestep indices.

        **kwargs
            Additional keyword arguments passed to the original model.

        Returns
        -------
        torch.Tensor
            Output of the original denoising network, evaluated at the mapped
            original timesteps.
        """
        # Convert the timestep map into a tensor on the same device as ts.
        map_tensor = th.tensor(
            self.timestep_map,
            device=ts.device,
            dtype=ts.dtype,
        )

        # Map reduced timestep indices to original diffusion timestep indices.
        new_ts = map_tensor[ts]

        # Optionally rescale timesteps to match the convention of the original
        # improved-diffusion implementation.
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        # Call the original PICDM denoising network with original timestep
        # embeddings and physical conditioning variables.
        return self.model(x, vxz, vel, loc, new_ts, **kwargs)
