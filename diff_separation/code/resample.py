from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a timestep schedule sampler for diffusion-model training.

    In PICDM training, each clean P-wave target

        x_0 = (Vx^p, Vz^p)

    is corrupted to a noisy P-wave mode x_t at a randomly selected diffusion
    timestep t. The denoising network is then trained to recover x_0 from x_t
    under the physical conditions:

        c = (Vx, Vz, vp, vs).

    This function chooses how the timestep t is sampled during training.

    Parameters
    ----------
    name : str
        Name of the schedule sampler. Supported options are:

            "uniform":
                Sample diffusion timesteps uniformly. This is the standard
                choice and is commonly used in DDPM/PICDM training.

            "loss-second-moment":
                Adaptively sample timesteps according to the second moment of
                recent training losses. Timesteps with larger losses are sampled
                more frequently to reduce training variance.

    diffusion : GaussianDiffusion
        Diffusion object containing the number of timesteps and related
        diffusion coefficients.

    Returns
    -------
    ScheduleSampler
        A timestep sampler used by the training loop.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for sampling diffusion timesteps during training.

    The diffusion training objective involves an expectation over timesteps t.
    Instead of evaluating the loss over all timesteps, each mini-batch samples
    a subset of timesteps. This class defines the sampling distribution over t.

    In PICDM, the sampled timestep determines the noise level added to the
    clean P-wave mode x_0. A small t corresponds to a lightly corrupted P-wave
    mode, while a large t corresponds to a highly noisy P-wave mode.

    By default, subclasses use unbiased importance sampling: even if timesteps
    are sampled non-uniformly, the returned weights correct the loss so that the
    expected objective remains unchanged.
    """

    @abstractmethod
    def weights(self):
        """
        Return unnormalized sampling weights for all diffusion timesteps.

        Returns
        -------
        np.ndarray
            Positive weights with shape [num_timesteps]. Larger weights mean
            the corresponding timesteps are sampled more frequently.
        """

    def sample(self, batch_size, device):
        """
        Sample diffusion timesteps for one mini-batch.

        Parameters
        ----------
        batch_size : int
            Number of training samples in the mini-batch. One timestep is
            sampled for each sample.

        device : torch.device
            Device on which the returned tensors should be placed.

        Returns
        -------
        indices : torch.Tensor
            Sampled timestep indices with shape [batch_size].

        weights : torch.Tensor
            Importance weights with shape [batch_size]. These weights are
            multiplied with the loss values in the training loop so that the
            timestep-resampled objective remains unbiased.

        Notes
        -----
        If p(t) is the probability of sampling timestep t, the corresponding
        importance weight is:

            1 / (T * p(t)),

        where T is the total number of diffusion timesteps.
        """
        # Get unnormalized weights and convert them into probabilities.
        w = self.weights()
        p = w / np.sum(w)

        # Sample timestep indices according to p(t).
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)

        # Importance weights preserve the unbiased training objective.
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)

        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniform timestep sampler.

    This sampler gives every diffusion timestep the same probability. It is the
    standard and simplest choice for diffusion training.

    In the PICDM training loop, UniformSampler means that the network learns to
    denoise P-wave modes across the full diffusion trajectory, from weakly noisy
    to nearly Gaussian-noise states.
    """

    def __init__(self, diffusion):
        """
        Initialize uniform weights over all diffusion timesteps.

        Parameters
        ----------
        diffusion : GaussianDiffusion
            Diffusion object. Its num_timesteps attribute determines the length
            of the sampling distribution.
        """
        self.diffusion = diffusion

        # Equal weight for every timestep.
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """
        Return uniform timestep weights.

        Returns
        -------
        np.ndarray
            Array of ones with shape [num_timesteps].
        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Base class for adaptive timestep samplers using observed training losses.

    Some diffusion timesteps may be harder to learn than others. For example,
    high-noise timesteps can be difficult because the P-wave mode is strongly
    corrupted, while low-noise timesteps may require fine details. A loss-aware
    sampler can increase the probability of sampling timesteps with larger
    recent losses.

    This class provides distributed synchronization logic. Subclasses define how
    the loss history is converted into sampling weights.
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update timestep reweighting using losses from the current worker.

        In distributed training, each rank observes losses for its own local
        mini-batch. This function gathers timestep-loss pairs from all ranks so
        that every worker updates its sampler with the same global information.

        Parameters
        ----------
        local_ts : torch.Tensor
            Timesteps sampled on the local worker.

        local_losses : torch.Tensor
            Loss values corresponding to local_ts. In PICDM, these losses can
            include both the data reconstruction loss and, if enabled, the
            physics-informed Laplacian residual loss.
        """
        # Gather the local batch size from every distributed worker.
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]

        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all gathered batches to the maximum batch size, because
        # torch.distributed.all_gather requires tensors with the same shape.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]

        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Remove padding and flatten gathered timestep-loss pairs.
        timesteps = [
            x.item()
            for y, bs in zip(timestep_batches, batch_sizes)
            for x in y[:bs]
        ]

        losses = [
            x.item()
            for y, bs in zip(loss_batches, batch_sizes)
            for x in y[:bs]
        ]

        # Delegate the actual update rule to the subclass.
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the sampler using globally gathered timestep-loss pairs.

        Subclasses should implement this method deterministically so that all
        distributed workers maintain identical sampler states.

        Parameters
        ----------
        ts : list[int]
            Diffusion timesteps.

        losses : list[float]
            Loss values corresponding to each timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Adaptive timestep sampler based on the second moment of recent losses.

    This sampler tracks a short history of losses for each diffusion timestep.
    Once enough history has been collected, it samples timesteps in proportion
    to the square root of the second moment of the loss:

        weight(t) ∝ sqrt(E[L_t^2]).

    As a result, timesteps with consistently larger losses are sampled more
    often. This can reduce gradient variance and focus training on more
    difficult noise levels.

    In PICDM, this can be useful if certain diffusion timesteps produce larger
    P-wave reconstruction errors or larger physics-informed residuals.
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        Initialize the loss-second-moment resampler.

        Parameters
        ----------
        diffusion : GaussianDiffusion
            Diffusion object containing the number of timesteps.

        history_per_term : int
            Number of recent loss values stored for each timestep.

        uniform_prob : float
            Small probability mass assigned uniformly to all timesteps. This
            prevents any timestep from having zero sampling probability and
            keeps the sampler exploratory.
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        # Loss history for each timestep.
        # Shape: [num_timesteps, history_per_term].
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term],
            dtype=np.float64,
        )

        # Number of recorded losses for each timestep.
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int64)

    def weights(self):
        """
        Return adaptive timestep weights.

        Before every timestep has accumulated enough history, use uniform
        weights. After warm-up, compute weights from the second moment of the
        loss history.

        Returns
        -------
        np.ndarray
            Sampling weights for all diffusion timesteps.
        """
        # Use uniform sampling until every timestep has enough history.
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)

        # Estimate sqrt(E[L_t^2]) for each timestep.
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))

        # Normalize weights to sum to one.
        weights /= np.sum(weights)

        # Mix with a small uniform distribution so all timesteps remain possible.
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)

        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Add newly observed losses to the per-timestep loss history.

        If a timestep already has a full history buffer, the oldest loss is
        removed and the new loss is appended. Otherwise, the new loss is stored
        in the next available position.

        Parameters
        ----------
        ts : list[int]
            Timesteps observed in the latest training mini-batches.

        losses : list[float]
            Loss values corresponding to those timesteps.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term and append the newest one.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # Fill the loss-history buffer during warm-up.
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check whether every timestep has enough loss-history entries.

        Returns
        -------
        bool
            True if every timestep has history_per_term recorded losses.
            Otherwise False.
        """
        return (self._loss_counts == self.history_per_term).all()
