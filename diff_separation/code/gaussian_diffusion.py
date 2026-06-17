import enum
import math

import numpy as np
import torch as th
import torch.nn.functional as F

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .datasets import denormalizer_vel


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Return a predefined beta schedule for the diffusion process.

    In the PICDM framework, the forward diffusion process gradually corrupts
    the clean P-wave mode

        x_0 = (Vx^p, Vz^p)

    into a noisy P-wave mode x_t by adding Gaussian noise over T timesteps.
    The beta schedule controls the amount of noise added at each timestep.

    Parameters
    ----------
    schedule_name : str
        Name of the noise schedule. Supported options are:
            - "linear": linear beta schedule from DDPM.
            - "cosine": cosine schedule commonly used in improved diffusion.

    num_diffusion_timesteps : int
        Number of diffusion timesteps T.

    Returns
    -------
    np.ndarray
        A 1-D array of beta values with length num_diffusion_timesteps.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al. The scale factor keeps the schedule
        # comparable when using a different number of diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        return np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )

    elif schedule_name == "cosine":
        # Cosine schedule used in improved diffusion. This is also the schedule
        # described in the training configuration of the paper.
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Construct beta values from a continuous alpha_bar function.

    alpha_bar(t) represents the cumulative product of alphas up to normalized
    diffusion time t. This function discretizes alpha_bar(t) into beta_t values.

    In the forward process,

        q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0,
                         (1 - alpha_bar_t) I),

    so alpha_bar_t determines how much of the clean P-wave mode remains at
    timestep t.

    Parameters
    ----------
    num_diffusion_timesteps : int
        Number of diffusion timesteps.

    alpha_bar : callable
        Function mapping t in [0, 1] to cumulative alpha_bar(t).

    max_beta : float
        Maximum beta value. This prevents numerical singularities.

    Returns
    -------
    np.ndarray
        Discretized beta schedule.
    """
    betas = []

    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps

        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Specify what the denoising network predicts.

    For PICDM, the most relevant option is START_X, where the network directly
    predicts the clean P-wave mode x_0 = (Vx^p, Vz^p). This x_0-prediction
    parameterization is convenient because the physics-informed loss is applied
    directly to the predicted P-wave components.
    """

    PREVIOUS_X = enum.auto()  # The model predicts x_{t-1}.
    START_X = enum.auto()    # The model predicts x_0.
    EPSILON = enum.auto()    # The model predicts the added Gaussian noise epsilon.


class ModelVarType(enum.Enum):
    """
    Specify how the reverse-process variance is modeled.

    The current PICDM implementation mainly focuses on the mean prediction,
    i.e., the generation of the P-wave mode. Variance options are retained from
    the original improved-diffusion implementation for compatibility.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    """
    Specify the diffusion training objective.

    For PICDM, MSE or RESCALED_MSE is typically used. The MSE term corresponds
    to the data-fidelity loss between the predicted clean P-wave mode and the
    reference P-wave mode obtained by the conventional separation method.
    """

    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        """
        Return whether the selected loss is a variational-bound objective.
        """
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Diffusion utilities for training and sampling PICDM.

    This class implements the major diffusion operations required by the
    physics-informed conditional diffusion model for elastic wave-mode separation:

        1. Forward diffusion:
               q(x_t | x_0)

           The clean P-wave mode x_0 = (Vx^p, Vz^p) is corrupted by Gaussian
           noise to produce x_t.

        2. Reverse denoising:
               p_theta(x_{t-1} | x_t, c)

           The neural network predicts the clean P-wave mode or an equivalent
           reverse-process quantity, conditioned on

               c = (Vx, Vz, vp, vs),

           where (Vx, Vz) are original coupled elastic wavefields and
           (vp, vs) are velocity models.

        3. Physics-informed training:
           If use_physicsloss is True, the loss includes a Laplacian residual
           enforcing the P-wave separation equation:

               Delta Vx^p = partial_xx Vx + partial_xz Vz,
               Delta Vz^p = partial_xz Vx + partial_zz Vz.

           In this implementation, the right-hand sides of these equations are
           precomputed and stored in mid_p.

        4. Physics-guided sampling:
           If pde_guide is True, each reverse sampling step is corrected by the
           gradient of the physics loss, pushing the generated P-wave mode toward
           physical consistency.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        use_physicsloss=False,
        pde_guide=False,
        fd_order=4,
    ):
        """
        Initialize the Gaussian diffusion process.

        Parameters
        ----------
        betas : np.ndarray
            Beta schedule controlling the forward noising process.

        model_mean_type : ModelMeanType
            Determines whether the network predicts x_{t-1}, x_0, or epsilon.
            For the PICDM paper, x_0 prediction is especially meaningful because
            the output directly represents the clean P-wave mode.

        model_var_type : ModelVarType
            Determines how the reverse-process variance is defined.

        loss_type : LossType
            Diffusion loss type. MSE-based losses are typically used for the
            P-wave reconstruction objective.

        rescale_timesteps : bool
            If True, rescale timesteps to the range used in the original
            1000-step diffusion implementation.

        use_physicsloss : bool
            If True, add the physics-informed Laplacian residual loss during
            training.

        pde_guide : bool
            If True, apply physics-guided correction during reverse sampling.

        fd_order : int
            Finite-difference order used for the Laplacian operator in the
            physics loss and physics guidance. The paper discusses high-order
            finite differences for accurate spatial derivatives.
        """
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Store beta schedule in float64 for numerical accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # alpha_t = 1 - beta_t.
        alphas = 1.0 - betas

        # alpha_bar_t = product_{s=1}^t alpha_s.
        # This quantity appears in the closed-form forward diffusion equation:
        #
        #   x_t = sqrt(alpha_bar_t) x_0
        #         + sqrt(1 - alpha_bar_t) epsilon.
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        # alpha_bar_{t-1} and alpha_bar_{t+1}.
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # Precompute terms used by q(x_t | x_0).
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)

        # Precompute terms used to recover x_0 from epsilon prediction.
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1
        )

        # Posterior q(x_{t-1} | x_t, x_0).
        # This is used to construct the reverse-process mean when the model
        # predicts x_0 or epsilon.
        self.posterior_variance = (
            betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        # Log variance is clipped because posterior variance is zero at t = 0.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        # Coefficients for the posterior mean:
        #
        #   q(x_{t-1} | x_t, x_0)
        #
        # The mean is a linear combination of x_0 and x_t.
        self.posterior_mean_coef1 = (
            betas
            * np.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # PICDM-specific options.
        self.use_physicsloss = use_physicsloss
        self.pde_guide = pde_guide
        self.fd_order = fd_order

        # Finite-difference Laplacian kernel used to compute:
        #
        #   Delta Vx^p and Delta Vz^p.
        #
        # These are compared with the precomputed right-hand-side terms stored
        # in mid_p.
        self.laplace_kernel = laplace_operator(order=fd_order)

    def q_mean_variance(self, x_start, t):
        """
        Compute q(x_t | x_0).

        Parameters
        ----------
        x_start : torch.Tensor
            Clean P-wave mode x_0 = (Vx^p, Vz^p).

        t : torch.Tensor
            Diffusion timestep indices.

        Returns
        -------
        mean : torch.Tensor
            Mean of q(x_t | x_0), equal to sqrt(alpha_bar_t) * x_0.

        variance : torch.Tensor
            Variance of q(x_t | x_0), equal to 1 - alpha_bar_t.

        log_variance : torch.Tensor
            Log variance of q(x_t | x_0).
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
        )

        variance = _extract_into_tensor(
            1.0 - self.alphas_cumprod,
            t,
            x_start.shape,
        )

        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample a noisy P-wave mode x_t from the clean P-wave mode x_0.

        This implements the closed-form forward diffusion equation used in the
        paper:

            x_t = sqrt(alpha_bar_t) x_0
                  + sqrt(1 - alpha_bar_t) epsilon,

        where epsilon is Gaussian noise.

        Parameters
        ----------
        x_start : torch.Tensor
            Clean P-wave mode x_0 = (Vx^p, Vz^p).

        t : torch.Tensor
            Diffusion timestep indices.

        noise : torch.Tensor, optional
            Gaussian noise. If None, random noise is generated.

        Returns
        -------
        torch.Tensor
            Noisy P-wave mode x_t.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape

        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_start.shape,
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the posterior q(x_{t-1} | x_t, x_0).

        This posterior is analytically available in DDPM. It is used to compute
        the reverse-process mean after the model predicts x_0.

        Parameters
        ----------
        x_start : torch.Tensor
            Predicted or true clean P-wave mode x_0.

        x_t : torch.Tensor
            Noisy P-wave mode at timestep t.

        t : torch.Tensor
            Diffusion timestep indices.

        Returns
        -------
        posterior_mean : torch.Tensor
            Mean of q(x_{t-1} | x_t, x_0).

        posterior_variance : torch.Tensor
            Variance of q(x_{t-1} | x_t, x_0).

        posterior_log_variance_clipped : torch.Tensor
            Clipped log variance.
        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
            * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
            * x_t
        )

        posterior_variance = _extract_into_tensor(
            self.posterior_variance,
            t,
            x_t.shape,
        )

        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped,
            t,
            x_t.shape,
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        x,
        vxz,
        vel,
        loc,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Compute the reverse-process distribution p_theta(x_{t-1} | x_t, c).

        This function applies the denoising network to the current noisy P-wave
        mode x_t and the physical conditioning variables.

        In PICDM, the network input contains:
            - x:   noisy P-wave mode x_t,
            - vxz: original elastic wavefield (Vx, Vz),
            - vel: velocity models (vp, vs),
            - loc: source/time metadata.

        The model output is interpreted according to model_mean_type. For
        START_X prediction, the model directly outputs the clean P-wave mode

            pred_xstart = (Vx^p, Vz^p).

        Parameters
        ----------
        model : torch.nn.Module
            Denoising network f_theta.

        x : torch.Tensor
            Current noisy P-wave mode x_t.

        vxz : torch.Tensor
            Original coupled elastic wavefield condition (Vx, Vz).

        vel : torch.Tensor
            Velocity condition (vp, vs).

        loc : torch.Tensor
            Source location and snapshot-time metadata.

        t : torch.Tensor
            Current diffusion timestep.

        clip_denoised : bool
            If True, clip predicted x_0 to [-1, 1].

        denoised_fn : callable, optional
            Optional function applied to predicted x_0 before clipping.

        model_kwargs : dict, optional
            Extra keyword arguments passed to the model.

        Returns
        -------
        dict
            Dictionary containing:
                mean:
                    Reverse-process mean.
                variance:
                    Reverse-process variance.
                log_variance:
                    Log variance.
                pred_xstart:
                    Predicted clean P-wave mode.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # Apply the conditional denoising network.
        # The network predicts either x_0, epsilon, or x_{t-1}, depending on
        # model_mean_type. In the paper, x_0 prediction is used to naturally
        # impose the physics loss on the clean P-wave mode.
        model_output = model(
            x,
            vxz,
            vel,
            loc,
            self._scale_timesteps(t),
            **model_kwargs,
        )

        # If the variance is learned, split the model output into mean-related
        # channels and variance-related channels.
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])

            model_output, model_var_values = th.split(model_output, C, dim=1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)

            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped,
                    t,
                    x.shape,
                )
                max_log = _extract_into_tensor(
                    np.log(self.betas),
                    t,
                    x.shape,
                )

                # model_var_values is in [-1, 1], interpolating between
                # the minimum and maximum log variance.
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)

        else:
            # Fixed variance options from the original DDPM implementation.
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])
                    ),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(
                model_log_variance,
                t,
                x.shape,
            )

        def process_xstart(x):
            """
            Optionally post-process and clip predicted x_0.
            """
            if denoised_fn is not None:
                x = denoised_fn(x)

            if clip_denoised:
                return x.clamp(-1, 1)

            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # Model predicts x_{t-1}; derive x_0 from x_{t-1}.
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(
                    x_t=x,
                    t=t,
                    xprev=model_output,
                )
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                # PICDM setting: directly predict clean P-wave mode x_0.
                pred_xstart = process_xstart(model_output)

            else:
                # If the model predicts epsilon, convert epsilon prediction to x_0.
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(
                        x_t=x,
                        t=t,
                        eps=model_output,
                    )
                )

            # Once x_0 is predicted, compute the reverse-process mean through
            # the analytical posterior q(x_{t-1} | x_t, x_0).
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart,
                x_t=x,
                t=t,
            )

        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape
            == model_log_variance.shape
            == pred_xstart.shape
            == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Recover x_0 from epsilon prediction.

        This is the rearranged version of the forward diffusion equation:

            x_t = sqrt(alpha_bar_t) x_0
                  + sqrt(1 - alpha_bar_t) epsilon.
        """
        assert x_t.shape == eps.shape

        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        Recover x_0 from a prediction of x_{t-1}.
        """
        assert x_t.shape == xprev.shape

        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                x_t.shape,
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Recover epsilon from x_t and predicted x_0.

        This is used in DDIM sampling, where the deterministic update requires
        the noise estimate implied by pred_xstart.
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        Optionally rescale diffusion timesteps before passing them to the model.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)

        return t

    def pde_loss(self, x, mid_p, dh):
        """
        Compute the physics-informed Laplacian loss for P-wave separation.

        In the paper, the predicted P-wave mode should satisfy Zhu's elastic
        wave-mode separation equations:

            Delta Vx^p = partial_xx Vx + partial_xz Vz,
            Delta Vz^p = partial_xz Vx + partial_zz Vz.

        In this implementation:
            - x[:, 0:1] is the predicted Vx^p,
            - x[:, 1:2] is the predicted Vz^p,
            - mid_p[:, 0:1] stores the precomputed right-hand side for Vx^p,
            - mid_p[:, 1:2] stores the precomputed right-hand side for Vz^p.

        Therefore, the loss penalizes the mismatch between:
            - Laplacian(predicted P-wave mode),
            - precomputed analytical separation terms.

        Parameters
        ----------
        x : torch.Tensor
            Predicted clean P-wave mode with two channels:
                x[:, 0] = Vx^p,
                x[:, 1] = Vz^p.

        mid_p : torch.Tensor
            Precomputed physical right-hand-side terms of the P-wave separation
            equations.

        dh : float
            Spatial grid interval. In the paper's experiments, dh = 10 m.

        Returns
        -------
        torch.Tensor
            Batch-wise physics-informed loss.
        """
        # Compute Delta Vx^p using a finite-difference Laplacian kernel.
        vxp_laplace = (
            1 / (dh ** 2)
            * F.conv2d(
                x[:, 0:1],
                self.laplace_kernel,
                padding=self.fd_order // 2,
                groups=1,
            )
        )

        # Compute Delta Vz^p.
        vzp_laplace = (
            1 / (dh ** 2)
            * F.conv2d(
                x[:, 1:2],
                self.laplace_kernel,
                padding=self.fd_order // 2,
                groups=1,
            )
        )

        # Remove boundary points affected by finite-difference padding.
        # This avoids including artificial boundary values in the physics loss.
        pad = self.fd_order // 2

        loss_vxp = mean_flat(
            (
                vxp_laplace[:, :, pad:-pad, pad:-pad]
                - mid_p[:, 0:1, pad:-pad, pad:-pad]
            )
            ** 2
        )

        loss_vzp = mean_flat(
            (
                vzp_laplace[:, :, pad:-pad, pad:-pad]
                - mid_p[:, 1:2, pad:-pad, pad:-pad]
            )
            ** 2
        )

        pde_loss = loss_vxp + loss_vzp

        return pde_loss

    def pde_guidance(self, x, mid_p, dh, weight_t, scale_factor=1.0):
        """
        Compute the physics-guidance gradient used during reverse sampling.

        This function implements the inference-stage physics correction described
        in the paper. After a denoising update produces an intermediate P-wave
        sample, the sample is further corrected using the gradient of the
        physics-informed loss:

            x <- x - eta * grad L_phys.

        Here, grad L_phys is computed by automatic differentiation. The gradient
        is normalized by its maximum absolute value for numerical stability.

        Parameters
        ----------
        x : torch.Tensor
            Current generated P-wave mode sample.

        mid_p : torch.Tensor
            Precomputed physical right-hand-side terms.

        dh : float
            Spatial grid spacing.

        weight_t : torch.Tensor
            Reverse-process variance or timestep-dependent weight.

        scale_factor : float
            Scaling factor controlling the strength of physics guidance.

        Returns
        -------
        cond_grad : torch.Tensor
            Physics-guidance correction term.

        pde_error : float
            Scalar diagnostic value of the physics residual.
        """
        with th.enable_grad():
            x.requires_grad_(True)

            vxp_laplace = (
                1 / (dh ** 2)
                * F.conv2d(
                    x[:, 0:1],
                    self.laplace_kernel,
                    padding=self.fd_order // 2,
                    groups=1,
                )
            )

            vzp_laplace = (
                1 / (dh ** 2)
                * F.conv2d(
                    x[:, 1:2],
                    self.laplace_kernel,
                    padding=self.fd_order // 2,
                    groups=1,
                )
            )

            loss_vxp = mean_flat((vxp_laplace - mid_p[:, 0:1]) ** 2)
            loss_vzp = mean_flat((vzp_laplace - mid_p[:, 1:2]) ** 2)

            pde_loss = (
                mean_flat((vxp_laplace - mid_p[:, 0:1]) ** 2)
                + mean_flat((vzp_laplace - mid_p[:, 1:2]) ** 2)
            )

            # Compute gradient of the physics loss with respect to the current
            # generated P-wave mode.
            grad = th.autograd.grad(pde_loss, x, retain_graph=True)[0]

            # Normalize the gradient to avoid unstable updates.
            grad_max = th.max(th.abs(grad)).item()
            grad = grad / grad_max

        # The correction is subtracted from the generated sample in p_sample or
        # ddim_sample. The factor 2 * pde_loss makes the guidance strength depend
        # on the current physics residual magnitude.
        return (
            weight_t * grad * scale_factor,
            th.sqrt((th.pow(loss_vxp, 2)).mean() + (th.pow(loss_vzp, 2)).mean()).item(),
        )

    def p_sample(
        self,
        model,
        x,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        t,
        scale_factor=1.0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Perform one stochastic DDPM reverse sampling step.

        This function samples x_{t-1} from p_theta(x_{t-1} | x_t, c), and then
        optionally applies physics-guided correction.

        The sampling step consists of:

            1. Denoising update:
                   use the neural network to estimate the reverse-process mean.

            2. Stochastic sampling:
                   add Gaussian noise according to the reverse variance.

            3. Physics correction:
                   if pde_guide is enabled, subtract the gradient of the
                   physics-informed loss to improve physical consistency.

        Returns
        -------
        dict
            sample:
                Updated P-wave sample x_{t-1}.

            pred_xstart:
                Network prediction of clean P-wave mode x_0.

            pde_loss_before:
                Physics residual before correction.

            pde_loss_after:
                Physics residual after correction.
        """
        out = self.p_mean_variance(
            model,
            x,
            vxz,
            vel,
            loc,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)

        # No random noise is added at t = 0 because the final output should be
        # the clean generated P-wave mode.
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        sample = (
            out["mean"]
            + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        )

        # Evaluate physical residual before correction.
        pde_loss_before = self.pde_loss(sample, mid_p, dh)

        pde_loss_after = 0.0

        if self.pde_guide:
            cond_grad, _ = self.pde_guidance(
                sample,
                mid_p,
                dh,
                out["variance"],
                scale_factor=scale_factor,
            )

            # Physics-guided correction:
            #   sample <- sample - eta * grad L_phys.
            sample = sample - cond_grad

            pde_loss_after = self.pde_loss(sample, mid_p, dh)

        if (t[0].item() + 1) % 100 == 0 or t[0].item() == 0:
            print(
                f"Time step {t[0].item()} --> "
                f"PDE guider before Loss {pde_loss_before.item()} "
                f"and after Loss {pde_loss_after.item()}"
            )

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "pde_loss_before": pde_loss_before.item(),
            "pde_loss_after": pde_loss_after.item(),
        }

    def p_sample_loop(
        self,
        model,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        shape,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Run the full DDPM reverse sampling chain.

        Starting from Gaussian noise x_T, this function repeatedly applies
        p_sample to obtain:

            x_T -> x_{T-1} -> ... -> x_0.

        The final sample is the generated P-wave mode.
        """
        final = None

        for sample, image_all, pred_xstart, pde_loss_before, pde_loss_after in (
            self.p_sample_loop_progressive(
                model,
                vxz,
                mid_p,
                vel,
                loc,
                dh,
                shape,
                scale_factor=scale_factor,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
            )
        ):
            final, image_all, pred_xstart, pde_loss_before, pde_loss_after = (
                sample,
                image_all,
                pred_xstart,
                pde_loss_before,
                pde_loss_after,
            )

        return final["sample"], image_all, pred_xstart, pde_loss_before, pde_loss_after

    def p_sample_loop_progressive(
        self,
        model,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        shape,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Progressive DDPM sampling generator.

        This yields intermediate samples at each reverse diffusion step. It is
        useful for visualizing how the generated P-wave mode evolves from pure
        Gaussian noise to a physically consistent separated P-wave field.
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)

        # Reverse diffusion timesteps: T-1, T-2, ..., 0.
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []
        pred_xstart = []
        pde_loss_before = []
        pde_loss_after = []

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    image,
                    vxz,
                    mid_p,
                    vel,
                    loc,
                    dh,
                    t,
                    scale_factor=scale_factor,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

                if (i + 1) % 50 == 0:
                    image_all.append(out["sample"])
                    pred_xstart.append(out["pred_xstart"])

                # NOTE:
                # The original code uses out["pred_loss_before"] and
                # out["pred_loss_after"], but p_sample returns keys named
                # "pde_loss_before" and "pde_loss_after". The corrected keys
                # are used below.
                pde_loss_before.append(out["pde_loss_before"])
                pde_loss_after.append(out["pde_loss_after"])

                yield out, image_all, pred_xstart, pde_loss_before, pde_loss_after

                image = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        t,
        scale_factor=1.0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Perform one DDIM reverse sampling step.

        DDIM provides a faster sampling alternative to standard DDPM. In the
        paper, DDIM is used to reduce the number of reverse steps, enabling
        efficient P-wave mode generation.

        The procedure is:

            1. Predict x_0 from x_t using the conditional denoising network.
            2. Convert predicted x_0 to the implied epsilon.
            3. Use the DDIM update to obtain x_{t-1}.
            4. Optionally apply physics-guided correction.

        Parameters
        ----------
        eta : float
            DDIM stochasticity parameter. eta = 0 gives deterministic DDIM.
        """
        out = self.p_mean_variance(
            model,
            x,
            vxz,
            vel,
            loc,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # Recover epsilon implied by the predicted clean P-wave mode.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)

        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = th.randn_like(x)

        # DDIM mean prediction for x_{t-1}.
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        sample = mean_pred + nonzero_mask * sigma * noise

        # Physics residual before guidance.
        pde_loss_before = self.pde_loss(sample, mid_p, dh).item()

        pde_loss_after = 0.0

        if self.pde_guide:
            cond_grad, _ = self.pde_guidance(
                sample,
                mid_p,
                dh,
                out["variance"],
                scale_factor=scale_factor,
            )

            # Apply physics-guided correction.
            sample = sample - cond_grad

            pde_loss_after = self.pde_loss(sample, mid_p, dh).item()

        if (t[0].item() + 1) % 5 == 0 or t[0].item() == 0:
            print(
                f"Time step {t[0].item()} --> "
                f"PDE guider before Loss {pde_loss_before} "
                f"and after Loss {pde_loss_after}"
            )

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "pde_loss_before": pde_loss_before,
            "pde_loss_after": pde_loss_after,
        }

    def ddim_sample_loop(
        self,
        model,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        shape,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Run the full DDIM reverse sampling process.

        This is the main accelerated inference routine for PICDM. It starts from
        Gaussian noise and iteratively generates the separated P-wave mode.
        """
        final = None

        for sample, image_all, pred_xstart, pde_loss_before, pde_loss_after in (
            self.ddim_sample_loop_progressive(
                model,
                vxz,
                mid_p,
                vel,
                loc,
                dh,
                shape,
                scale_factor=scale_factor,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
            )
        ):
            final, image_all, pred_xstart, pde_loss_before, pde_loss_after = (
                sample,
                image_all,
                pred_xstart,
                pde_loss_before,
                pde_loss_after,
            )

        return final["sample"], image_all, pred_xstart, pde_loss_before, pde_loss_after

    def ddim_sample_loop_progressive(
        self,
        model,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        shape,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Progressive DDIM sampling generator.

        This function yields intermediate DDIM samples. It can be used to
        monitor the reverse process and to record the evolution of the generated
        P-wave mode and the physics residual.
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []
        pred_xstart = []
        pde_loss_before = []
        pde_loss_after = []

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    image,
                    vxz,
                    mid_p,
                    vel,
                    loc,
                    dh,
                    t,
                    scale_factor=scale_factor,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )

            if i % 50 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])

            pde_loss_before.append(out["pde_loss_before"])
            pde_loss_after.append(out["pde_loss_after"])

            yield out, image_all, pred_xstart, pde_loss_before, pde_loss_after

            image = out["sample"]

    def _vb_terms_bpd(
        self,
        model,
        x_start,
        x_t,
        vxz,
        vel,
        loc,
        t,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Compute variational-bound terms in bits per dimension.

        This function is inherited from the original diffusion implementation.
        It is mainly used when the loss type is KL-based or when learning the
        reverse-process variance.

        For the main PICDM training with MSE and physics-informed loss, this is
        not the central objective.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start,
            x_t=x_t,
            t=t,
        )

        out = self.p_mean_variance(
            model,
            x_t,
            vxz,
            vel,
            loc,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        kl = normal_kl(
            true_mean,
            true_log_variance_clipped,
            out["mean"],
            out["log_variance"],
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=0.5 * out["log_variance"],
        )

        assert decoder_nll.shape == x_start.shape

        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At t = 0, use decoder negative log likelihood.
        # Otherwise, use the KL term.
        output = th.where((t == 0), decoder_nll, kl)

        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model,
        x_start,
        vxz,
        mid_p,
        vel,
        loc,
        dh,
        t,
        model_kwargs=None,
        noise=None,
    ):
        """
        Compute the PICDM training loss at sampled diffusion timesteps.

        This is the main training objective used by train_util.py.

        In the paper's notation:

            x_start = x_0 = (Vx^p, Vz^p),

        where x_0 is the clean P-wave mode. The function first constructs a
        noisy P-wave mode x_t through forward diffusion:

            x_t = sqrt(alpha_bar_t) x_0
                  + sqrt(1 - alpha_bar_t) epsilon.

        The denoising network then predicts the target according to
        model_mean_type. For START_X prediction, the target is simply x_start.

        If use_physicsloss is True, the physics-informed loss is added:

            L_train = L_data + 100 * L_phys,

        where L_phys enforces the Laplacian P-wave separation equation. The
        factor 100 corresponds to the lambda value used in the paper.

        Parameters
        ----------
        model : torch.nn.Module
            Conditional denoising network.

        x_start : torch.Tensor
            Clean P-wave target x_0 = (Vx^p, Vz^p).

        vxz : torch.Tensor
            Original elastic wavefield condition (Vx, Vz).

        mid_p : torch.Tensor
            Precomputed right-hand side of the P-wave separation equation.

        vel : torch.Tensor
            Velocity condition (vp, vs).

        loc : torch.Tensor
            Source/time metadata.

        dh : float
            Spatial grid spacing for finite-difference derivatives.

        t : torch.Tensor
            Sampled diffusion timesteps.

        model_kwargs : dict, optional
            Additional model conditions.

        noise : torch.Tensor, optional
            Gaussian noise used in q_sample. If None, random noise is generated.

        Returns
        -------
        dict
            Loss dictionary. Common keys include:
                - mse: data reconstruction loss,
                - ploss: physics-informed loss,
                - loss: total training loss.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = th.randn_like(x_start)

        # Forward diffusion: corrupt the clean P-wave mode x_0 into x_t.
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                vxz=vxz,
                vel=vel,
                loc=loc,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # Apply the denoising network to predict x_0, epsilon, or x_{t-1}.
            model_output = model(
                x_t,
                vxz,
                vel,
                loc,
                self._scale_timesteps(t),
                **model_kwargs,
            )

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]

                assert model_output.shape == (B, C * 2, *x_t.shape[2:])

                model_output, model_var_values = th.split(model_output, C, dim=1)

                # Learn the variance using the variational bound, but stop the
                # variance loss from affecting the mean prediction.
                frozen_out = th.cat(
                    [model_output.detach(), model_var_values],
                    dim=1,
                )

                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    vxz=vxz,
                    vel=vel,
                    loc=loc,
                    t=t,
                    clip_denoised=False,
                )["output"]

                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            # Define the supervised diffusion target.
            # For PICDM with START_X prediction:
            #
            #     target = x_start = clean P-wave mode.
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            # Data-fidelity loss L_data. For START_X prediction, this measures
            # the MSE between predicted and reference P-wave modes.
            terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

            # Physics-informed loss L_phys.
            # This enforces the Laplacian relation in the P-wave separation
            # equation and corresponds to the physics loss in the paper.
            if self.use_physicsloss:
                terms["ploss"] = self.pde_loss(model_output, mid_p, dh)

                # Lambda = 100, matching the training configuration described
                # in the paper.
                terms["loss"] += 100 * terms["ploss"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Compute the prior KL term for the variational lower bound.

        This term depends only on the forward diffusion process and is not
        optimized directly.
        """
        batch_size = x_start.shape[0]

        t = th.tensor(
            [self.num_timesteps - 1] * batch_size,
            device=x_start.device,
        )

        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)

        kl_prior = normal_kl(
            mean1=qt_mean,
            logvar1=qt_log_variance,
            mean2=0.0,
            logvar2=0.0,
        )

        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(
        self,
        model,
        x_start,
        vxz,
        vel,
        loc,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Compute the complete variational lower bound over all timesteps.

        This diagnostic routine is inherited from standard diffusion models.
        It is not the main metric used for the PICDM wave-mode separation
        experiments, where MSE/NMSE against the conventional separation result
        and physics residuals are more directly relevant.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []

        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)

            x_t = self.q_sample(
                x_start=x_start,
                t=t_batch,
                noise=noise,
            )

            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    vxz=vxz,
                    vel=vel,
                    loc=loc,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )

            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))

            eps = self._predict_eps_from_xstart(
                x_t,
                t_batch,
                out["pred_xstart"],
            )
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract timestep-dependent coefficients and broadcast them to a tensor shape.

    Diffusion coefficients such as sqrt(alpha_bar_t) are stored as 1-D NumPy
    arrays over timesteps. This helper selects the coefficients corresponding
    to the batch timesteps and expands them to match the wavefield tensor shape.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of diffusion coefficients.

    timesteps : torch.Tensor
        Batch of timestep indices.

    broadcast_shape : tuple
        Target tensor shape, usually [batch, channels, nz, nx].

    Returns
    -------
    torch.Tensor
        Broadcasted coefficient tensor.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res.expand(broadcast_shape)


def laplace_operator(order=4, device="cuda"):
    """
    Construct a finite-difference Laplacian convolution kernel.

    The physics-informed loss requires the Laplacian of the predicted P-wave
    components:

        Delta Vx^p and Delta Vz^p.

    This function returns a 2D convolution kernel approximating the Laplacian
    operator with a selected finite-difference order.

    Parameters
    ----------
    order : int
        Finite-difference order. Supported values:
            - 2: second-order Laplacian stencil,
            - 4: fourth-order Laplacian stencil,
            - 8: eighth-order Laplacian stencil.

    device : str
        Device on which the kernel is allocated.

    Returns
    -------
    torch.Tensor
        Laplacian kernel with shape [1, 1, k, k].
    """
    if order == 2:
        laplace_kernel = th.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=th.float32,
        )

        laplace_kernel = laplace_kernel.view(1, 1, 3, 3).to(device)

    elif order == 4:
        laplace_kernel = th.tensor(
            [
                [0, 0, -1 / 12, 0, 0],
                [0, 0, 16 / 12, 0, 0],
                [-1 / 12, 16 / 12, -30 / 12 * 2, 16 / 12, -1 / 12],
                [0, 0, 16 / 12, 0, 0],
                [0, 0, -1 / 12, 0, 0],
            ],
            dtype=th.float32,
        )

        laplace_kernel = laplace_kernel.view(1, 1, 5, 5).to(device)

    elif order == 8:
        laplace_kernel = th.tensor(
            [
                [
                    [
                        [0, 0, 0, 0, -1 / 560, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8 / 315, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1 / 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8 / 5, 0, 0, 0, 0],
                        [
                            -1 / 560,
                            8 / 315,
                            -1 / 5,
                            8 / 5,
                            -205 * 2 / 72,
                            8 / 5,
                            -1 / 5,
                            8 / 315,
                            -1 / 560,
                        ],
                        [0, 0, 0, 0, 8 / 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1 / 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 8 / 315, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1 / 560, 0, 0, 0, 0],
                    ]
                ]
            ],
            dtype=th.float32,
            device=device,
        )

    else:
        raise NotImplementedError(f"Unsupported finite-difference order: {order}")

    return laplace_kernel
