import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel


# Number of classes used only when class-conditional training is enabled.
# In the current PICDM setting, class_cond=False by default because the model
# is conditioned on physical quantities rather than semantic class labels.
NUM_CLASSES = 4


def model_and_diffusion_defaults():
    """
    Return default configuration for the PICDM model and diffusion process.

    These defaults define both the conditional U-Net architecture and the
    Gaussian diffusion process used for elastic P-wave mode separation.

    In the PICDM paper, the model learns to generate the clean P-wave mode

        x_0 = (Vx^p, Vz^p)

    from a noisy P-wave mode x_t, conditioned on the original elastic wavefield
    and velocity models:

        c = (Vx, Vz, vp, vs).

    Important default settings
    --------------------------
    in_channels = 6
        The network input has six channels:
            1-2: noisy P-wave mode x_t = (Vx^p, Vz^p) at timestep t,
            3-4: original coupled elastic wavefield (Vx, Vz),
            5-6: velocity models (vp, vs).

    out_channels = 2
        The network predicts two channels corresponding to the clean P-wave
        mode:
            output[0] = Vx^p,
            output[1] = Vz^p.

    diffusion_steps = 1000
        The original diffusion process contains 1000 timesteps. During training,
        the clean P-wave target x_0 is randomly noised to x_t using these
        timesteps.

    noise_schedule = "cosine"
        Cosine noise schedule used to define the beta values for the diffusion
        process.

    timestep_respacing = ""
        This is important.

        During training, timestep_respacing is set to an empty string "".
        In create_gaussian_diffusion(), this empty value is converted to
        [steps], so the model uses the full 1000-step diffusion process.

        During sampling/inference, this can be changed to values such as
        "ddim50". In that case, only 50 timesteps are selected from the original
        1000-step schedule, and the model can perform accelerated DDIM sampling.

    predict_xstart = True
        The model directly predicts x_0, i.e., the clean P-wave mode. This
        matches the paper's x_0-prediction parameterization and makes it
        straightforward to apply the physics-informed loss on the predicted
        P-wave components.

    use_physicsloss = True
        Add the physics-informed Laplacian loss during training. This implements
        the physics constraint in the PICDM objective:

            L_train = L_data + lambda L_phys.

    pde_guide = False
        Disable physics-guided sampling by default. During inference, this can
        be set to True to apply gradient-based physics correction at each
        reverse diffusion step.

    fd_order = 8
        Use an eighth-order finite-difference Laplacian operator for the
        physics-informed loss and physics-guided correction.
    """
    return dict(
        # Model input/output settings.
        in_channels=6,
        num_channels=64,
        out_channels=2,

        # Five-scale U-Net channel multiplier.
        # With num_channels=64, the feature widths are:
        # 64, 128, 256, 512, and 1024.
        channel_mult=(1, 2, 4, 8, 16),

        # Two residual blocks at each U-Net scale.
        num_res_blocks=2,

        # Multi-head attention settings.
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions=(8, 16, 32),

        # Dropout is disabled for deterministic wavefield reconstruction.
        dropout=0.0,

        # Diffusion variance and class-conditioning settings.
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,

        # Diffusion process settings.
        diffusion_steps=1000,
        noise_schedule="cosine",

        # Training: keep "" to use the full 1000 diffusion steps.
        # Sampling: set to "ddim50" to use 50-step DDIM accelerated sampling.
        timestep_respacing="",

        # Loss and prediction parameterization.
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,

        # U-Net implementation options.
        use_checkpoint=False,
        use_scale_shift_norm=True,

        # PICDM physics-informed options.
        use_physicsloss=True,
        pde_guide=False,
        fd_order=8,
    )


def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_physicsloss,
    pde_guide,
    fd_order,
):
    """
    Create both the PICDM denoising network and the diffusion process.

    This function is a high-level factory used by training and inference scripts.
    It first builds the conditional U-Net model and then constructs the Gaussian
    diffusion object.

    Parameters
    ----------
    class_cond : bool
        Whether to use class conditioning. For PICDM, this is normally False
        because the model is physically conditioned on wavefields and velocity
        models rather than class labels.

    learn_sigma : bool
        Whether the model also predicts the reverse-process variance.

    sigma_small : bool
        Whether to use the smaller fixed variance option when learn_sigma=False.

    in_channels : int
        Number of model input channels. For PICDM, this is 6:
            x_t has 2 channels,
            (Vx, Vz) has 2 channels,
            (vp, vs) has 2 channels.

    num_channels : int
        Base U-Net channel width.

    out_channels : int
        Number of output channels. For PICDM, this is 2, corresponding to
        predicted clean P-wave components (Vx^p, Vz^p).

    channel_mult : tuple[int]
        Channel multiplier for each U-Net scale.

    num_res_blocks : int
        Number of residual blocks at each U-Net scale.

    num_heads : int
        Number of attention heads.

    num_heads_upsample : int
        Number of attention heads used in upsampling blocks.

    attention_resolutions : tuple[int]
        Spatial resolutions where self-attention is applied.

    dropout : float
        Dropout probability.

    diffusion_steps : int
        Number of original diffusion timesteps. In the paper, this is 1000.

    noise_schedule : str
        Beta schedule type, e.g., "cosine".

    timestep_respacing : str or list
        Controls whether the original diffusion process is respaced.

        Training setting:
            timestep_respacing = ""

            An empty string means no acceleration. The code automatically
            converts it to [diffusion_steps], which keeps all 1000 timesteps.

        Sampling setting used in the paper:
            timestep_respacing = "ddim50"

            This selects 50 timesteps from the original 1000-step diffusion
            process. When combined with ddim_sample_loop(), the model performs
            50-step DDIM accelerated sampling.

    use_kl : bool
        Whether to use a KL-based variational objective.

    predict_xstart : bool
        If True, the model predicts x_0 directly. For PICDM, this should be
        True because x_0 is the clean P-wave mode and the physics loss is applied
        directly to this prediction.

    rescale_timesteps : bool
        Whether to rescale timesteps before passing them to the model.

    rescale_learned_sigmas : bool
        Whether to use the rescaled MSE objective for learned variances.

    use_checkpoint : bool
        Whether to use gradient checkpointing in the U-Net.

    use_scale_shift_norm : bool
        Whether to use scale-shift normalization in residual blocks.

    use_physicsloss : bool
        Whether to add the physics-informed loss during training.

    pde_guide : bool
        Whether to enable physics-guided correction during sampling.

    fd_order : int
        Finite-difference order used in the Laplacian operator.

    Returns
    -------
    model : UNetModel
        Conditional denoising network.

    diffusion : SpacedDiffusion
        Gaussian diffusion process, possibly respaced for accelerated sampling.
    """
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_physicsloss=use_physicsloss,
        pde_guide=pde_guide,
        fd_order=fd_order,
    )

    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Create the conditional U-Net denoising network used by PICDM.

    The model receives the noisy P-wave mode x_t together with physical
    conditioning variables and predicts the clean P-wave mode x_0.

    In this implementation, the input tensor has six channels and the output
    tensor has two channels:

        input  = concat(x_t, Vx, Vz, vp, vs)
        output = predicted x_0 = (Vx^p, Vz^p)

    The exact concatenation is handled inside the U-Net forward function.
    This factory only defines the architecture.
    """
    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    use_physicsloss=False,
    pde_guide=False,
    fd_order=4,
):
    """
    Create the Gaussian/Spaced diffusion process used by PICDM.

    This function defines the diffusion noise schedule, the training loss type,
    the prediction target, the variance type, and the timestep respacing strategy.

    The most important argument here is timestep_respacing.

    Training setting
    ----------------
    During training, set:

        timestep_respacing = ""

    If timestep_respacing is empty, the code executes:

        timestep_respacing = [steps]

    Since steps=1000 by default, this means:

        timestep_respacing = [1000]

    Then space_timesteps(1000, [1000]) returns all original 1000 timesteps.
    Therefore, the model is trained with the complete 1000-step diffusion
    process. At each iteration, a timestep t is sampled from these 1000 steps,
    and the clean P-wave mode x_0 is noised into x_t.

    Sampling setting in the paper
    -----------------------------
    During inference/sampling, set:

        timestep_respacing = "ddim50"

    This asks space_timesteps(...) to select 50 timesteps from the original
    1000-step schedule. The returned SpacedDiffusion object therefore has only
    50 effective reverse steps. When this respaced diffusion object is used with
    ddim_sample_loop(...), the model performs 50-step DDIM accelerated sampling.

    In other words:

        training: timestep_respacing=""       -> full 1000-step diffusion
        sampling: timestep_respacing="ddim50" -> 50-step DDIM sampling

    This design allows the model to learn from the full diffusion trajectory
    during training while using much fewer steps for efficient inference.

    Parameters
    ----------
    steps : int
        Number of timesteps in the original diffusion process. The default is
        1000, matching the paper's training setting.

    learn_sigma : bool
        Whether the model learns reverse-process variance.

    sigma_small : bool
        If learn_sigma=False, choose whether to use the small fixed variance.

    noise_schedule : str
        Name of the beta schedule, e.g., "linear" or "cosine".

    use_kl : bool
        If True, use the variational KL objective.

    predict_xstart : bool
        If True, the network predicts x_0 directly. For PICDM, this is the
        clean P-wave mode (Vx^p, Vz^p).

    rescale_timesteps : bool
        If True, rescale timestep embeddings to the conventional 0--1000 range.

    rescale_learned_sigmas : bool
        Whether to use the rescaled MSE objective for learned variances.

    timestep_respacing : str or list
        Respacing rule for selecting a subset of timesteps. Use "" for full
        training and "ddim50" for 50-step DDIM accelerated sampling.

    use_physicsloss : bool
        Whether to add the physics-informed Laplacian loss during training.

    pde_guide : bool
        Whether to use physics-guided correction during reverse sampling.

    fd_order : int
        Finite-difference order used for the Laplacian operator in the physics
        loss and physics-guided sampling.

    Returns
    -------
    SpacedDiffusion
        Diffusion object using either the full timestep schedule or a respaced
        schedule.
    """
    # Construct beta_t for the original diffusion process.
    # For the paper's main setting, steps=1000 and noise_schedule="cosine".
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Select the loss type.
    # For the PICDM main setting, use_kl=False and rescale_learned_sigmas=False,
    # so the training objective is MSE, optionally plus the physics-informed loss.
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # Important timestep-respacing logic.
    #
    # If timestep_respacing is empty, use all original diffusion steps.
    # Therefore, during training:
    #
    #     timestep_respacing = ""
    #     steps = 1000
    #
    # becomes:
    #
    #     timestep_respacing = [1000]
    #
    # and the diffusion process retains all 1000 timesteps.
    #
    # During sampling, set:
    #
    #     timestep_respacing = "ddim50"
    #
    # so space_timesteps(...) selects 50 timesteps from the original 1000-step
    # process. Combined with ddim_sample_loop(...), this gives 50-step DDIM
    # accelerated sampling, as used in the paper.
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,

        # Prediction target.
        # For PICDM, predict_xstart=True, so the model predicts x_0 directly:
        #     x_0 = (Vx^p, Vz^p).
        model_mean_type=(
            gd.ModelMeanType.EPSILON
            if not predict_xstart
            else gd.ModelMeanType.START_X
        ),

        # Reverse-process variance setting.
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),

        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,

        # PICDM-specific physics options.
        use_physicsloss=use_physicsloss,
        pde_guide=pde_guide,
        fd_order=fd_order,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add configuration entries to an argparse parser.

    This utility converts the default dictionary returned by
    model_and_diffusion_defaults() into command-line arguments.

    For example, the following default entry:

        timestep_respacing=""

    becomes the command-line argument:

        --timestep_respacing ""

    During sampling, one can override it as:

        --timestep_respacing ddim50

    to enable 50-step DDIM accelerated sampling.
    """
    for k, v in default_dict.items():
        v_type = type(v)

        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool

        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Convert selected argparse arguments into a dictionary.

    This is used to pass only the relevant command-line arguments into
    create_model_and_diffusion() or create_gaussian_diffusion().
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Convert common string representations to Boolean values.

    This allows command-line arguments such as:

        --use_physicsloss True
        --pde_guide False

    to be parsed correctly as Python booleans.
    """
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise argparse.ArgumentTypeError("boolean value expected")
