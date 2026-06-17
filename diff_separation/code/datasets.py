import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random

def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    Create an infinite data loader for the PICDM training samples.

    In the proposed physics-informed conditional diffusion model (PICDM),
    each training sample contains:

        1. vxz_p:
           The target clean P-wave mode x_0 = (Vx^p, Vz^p).
           This is the ground-truth label used in the data-fidelity loss.

        2. vxz:
           The original coupled elastic velocity wavefield (Vx, Vz).
           This is part of the conditioning input c in the paper.

        3. mid_p:
           An auxiliary/intermediate P-wave mode representation.
           Depending on the training or sampling script, this may be used
           as an additional P-wave-related input or reference.

        4. vel:
           The normalized P- and S-wave velocity models (vp, vs).
           These are also part of the conditioning input c = (Vx, Vz, vp, vs).

        5. loc:
           Normalized source location and snapshot-time information:
           (sx, sz, snapit). These variables provide acquisition/time metadata.

        6. out_dict:
           A placeholder dictionary kept for compatibility with the original
           improved-diffusion data-loading interface.

    The returned loader yields batches indefinitely. This matches the training
    style of diffusion models, where the training loop repeatedly samples
    mini-batches and randomly selects diffusion timesteps.

    Parameters
    ----------
    data_dir : str
        Directory containing the training files. Each file is expected to be
        a .npz, .npy, or .mat file. In this implementation, samples are loaded
        with np.load, so .npz files are typically expected.

    batch_size : int
        Number of samples in each mini-batch.

    class_cond : bool, optional
        Kept for compatibility with class-conditional diffusion code.
        It is not used in the current PICDM setting because the conditioning
        is physical, not class-based.

    deterministic : bool, optional
        If True, samples are loaded in a fixed order without shuffling.
        This is useful for validation or reproducibility checks.
        If False, samples are shuffled during training.

    Yields
    ------
    batch
        A batch returned by BasicDataset, containing:
        vxz_p, vxz, mid_p, vel, loc, out_dict.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Recursively collect all supported data files from the dataset directory.
    all_files = _list_image_files_recursively(data_dir)

    # Construct the dataset object. Each item corresponds to one elastic
    # wavefield snapshot and its associated P-wave label and conditioning fields.
    dataset = BasicDataset(
        all_files,
        class_cond=class_cond,
    )

    # For deterministic evaluation, disable shuffling.
    # For training, shuffle the samples to improve stochastic optimization.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=4,
        drop_last=True,
    )

    # Yield batches forever. This follows the common diffusion-model training
    # convention where the outer training loop controls the number of iterations.
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively list all supported data files in a directory.

    Although the function name contains "image", here the files are not images.
    They are seismic wavefield training samples stored in .npz/.npy/.mat format.

    Each file is expected to contain elastic wavefield components, separated
    P-wave components, velocity models, and acquisition/time metadata.

    Parameters
    ----------
    data_dir : str
        Root directory of the dataset.

    Returns
    -------
    results : list of str
        Sorted list of file paths ending with .mat, .npz, or .npy.
    """
    results = []

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]

        # Supported sample formats.
        # In the current __getitem__ implementation, np.load(path) is used,
        # so .npz/.npy files are directly supported.
        if "." in entry and ext.lower() in ["mat", "npz", "npy"]:
            results.append(full_path)

        # Recursively search subdirectories.
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))

    return results


def normalizer_vp(x, dmin=1500, dmax=5000):
    """
    Normalize the P-wave velocity model vp to the range [-1, 1].

    The paper uses vp as one of the conditional inputs c = (Vx, Vz, vp, vs).
    Normalization helps stabilize neural-network training by keeping different
    physical quantities within comparable numerical ranges.

    Parameters
    ----------
    x : array-like
        P-wave velocity model, usually in m/s.

    dmin : float
        Minimum vp value used for normalization.

    dmax : float
        Maximum vp value used for normalization.

    Returns
    -------
    array-like
        Normalized vp in the range approximately [-1, 1].
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def normalizer_vs(x, dmin=780, dmax=3500):
    """
    Normalize the S-wave velocity model vs to the range [-1, 1].

    The S-wave velocity model is used as part of the physical conditioning
    information. Although the analytical separation equation does not explicitly
    contain vp or vs, the paper shows that velocity conditioning improves
    generalization, especially for out-of-distribution velocity structures.

    Parameters
    ----------
    x : array-like
        S-wave velocity model, usually in m/s.

    dmin : float
        Minimum vs value used for normalization.

    dmax : float
        Maximum vs value used for normalization.

    Returns
    -------
    array-like
        Normalized vs in the range approximately [-1, 1].
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def normalizer_it(x, dmin=0, dmax=1500):
    """
    Normalize the snapshot time index to the range [-1, 1].

    In the training dataset described in the paper, elastic wavefield snapshots
    are sampled from the simulated time window. The variable snapit records the
    time index of the selected snapshot. Normalizing it provides compact metadata
    that can be used by downstream scripts if time/location conditioning is enabled.

    Parameters
    ----------
    x : array-like or scalar
        Snapshot index.

    dmin : float
        Minimum snapshot index.

    dmax : float
        Maximum snapshot index, corresponding to the 1.5 s training window
        when the time step is 1 ms.

    Returns
    -------
    array-like or scalar
        Normalized snapshot index.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def normalizer_coords(x, dmin=0, dmax=256):
    """
    Normalize source coordinates to the range [-1, 1].

    The training samples are generated on 256 x 256 grids. Source coordinates
    sx and sz are normalized so that acquisition geometry information can be
    represented on a scale compatible with neural-network inputs.

    Parameters
    ----------
    x : array-like or scalar
        Source coordinate along x or z.

    dmin : float
        Minimum grid coordinate.

    dmax : float
        Maximum grid coordinate for the 256 x 256 training models.

    Returns
    -------
    array-like or scalar
        Normalized coordinate.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def denormalizer_vel(x, dmin=1.5, dmax=4.5):
    """
    Convert a normalized velocity field from [-1, 1] back to physical units.

    This helper is useful when visualizing or post-processing normalized velocity
    models. The default range assumes velocity in km/s rather than m/s.

    Parameters
    ----------
    x : array-like
        Normalized velocity field.

    dmin : float
        Minimum physical velocity value.

    dmax : float
        Maximum physical velocity value.

    Returns
    -------
    array-like
        Denormalized velocity field.
    """
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


class BasicDataset(Dataset):
    """
    Dataset class for loading PICDM elastic wavefield separation samples.

    Each sample corresponds to one elastic wavefield snapshot. The loaded data
    are organized according to the PICDM formulation:

        Target:
            vxz_p = (Vx^p, Vz^p)

        Physical condition:
            vxz = (Vx, Vz)
            vel = (vp, vs)

        Optional metadata:
            loc = (sx, sz, snapit)

    During diffusion training, the clean target vxz_p is further corrupted by
    the forward diffusion process outside this dataset class:

        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon,

    where x_0 is vxz_p. The denoising network then learns to recover the clean
    P-wave mode from the noisy P-wave mode x_t under the physical conditions.
    """

    def __init__(self, paths, class_cond=False):
        """
        Initialize the dataset.

        Parameters
        ----------
        paths : list of str
            Paths to the wavefield sample files.

        class_cond : bool, optional
            Placeholder for compatibility with class-conditional diffusion code.
            This is not used in the current physics-conditioned PICDM.
        """
        super().__init__()
        self.local_dataset = paths
        self.class_cond = class_cond

    def __len__(self):
        """
        Return the number of available wavefield samples.
        """
        return len(self.local_dataset)

    def __getitem__(self, idx):
        """
        Load and organize one PICDM training sample.

        The expected file contains the following arrays:

            vx, vz:
                Original coupled elastic velocity wavefield components.

            vx_p, vz_p:
                Reference P-wave components obtained by the conventional
                numerical wave-mode separation method. These are the clean
                targets x_0 for diffusion-model training.

            mid_vxp, mid_vzp:
                Auxiliary/intermediate P-wave components. These can be used
                by other scripts as intermediate physical references or
                additional P-wave-related inputs.

            vp, vs:
                P- and S-wave velocity models. These are normalized and used
                as velocity conditioning.

            sx, sz:
                Source coordinates.

            snapit:
                Snapshot time index.

        Returns
        -------
        vxz_p : np.ndarray, shape (2, nz, nx)
            Clean P-wave target x_0 = (Vx^p, Vz^p).

        vxz : np.ndarray, shape (2, nz, nx)
            Original elastic wavefield condition (Vx, Vz).

        mid_p : np.ndarray, shape (2, nz, nx)
            Auxiliary/intermediate P-wave field.

        vel : np.ndarray, shape (2, nz, nx)
            Normalized velocity condition (vp, vs).

        loc : np.ndarray, shape (3, ...)
            Normalized source/time metadata (sx, sz, snapit).

        out_dict : dict
            Empty dictionary kept for compatibility with the improved-diffusion
            training interface.
        """
        path = self.local_dataset[idx]

        # Load one wavefield sample.
        # The sample is expected to be a NumPy archive containing all variables
        # required for supervised and physics-informed diffusion training.
        dict = np.load(path)

        # Original coupled elastic velocity wavefield components.
        # These correspond to (Vx, Vz) in the paper.
        vx = dict["vx"]
        vz = dict["vz"]

        # Clean separated P-wave components.
        # These correspond to the target x_0 = (Vx^p, Vz^p).
        vx_p = dict["vx_p"]
        vz_p = dict["vz_p"]

        # Auxiliary/intermediate P-wave components.
        # Their exact role depends on the downstream training or inference script.
        mid_vxp = dict["mid_vxp"]
        mid_vzp = dict["mid_vzp"]

        # P- and S-wave velocity models used as conditioning information.
        # These are normalized before being passed to the network.
        vp = dict["vp"]
        vs = dict["vs"]
        vp = normalizer_vp(vp)
        vs = normalizer_vs(vs)

        # Source location and snapshot-time metadata.
        sx = dict["sx"]
        sz = dict["sz"]
        snapit = dict["snapit"]

        # Normalize coordinates and snapshot index to improve numerical stability.
        sx = normalizer_coords(sx)
        sz = normalizer_coords(sz)
        snapit = normalizer_it(snapit)

        # Convert all wavefield arrays to float32 for efficient PyTorch training.
        vx = np.array(vx, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        vx_p = np.array(vx_p, dtype=np.float32)
        vz_p = np.array(vz_p, dtype=np.float32)
        mid_vxp = np.array(mid_vxp, dtype=np.float32)
        mid_vzp = np.array(mid_vzp, dtype=np.float32)

        # Convert normalized velocity models to float32.
        vp = np.array(vp, dtype=np.float32)
        vs = np.array(vs, dtype=np.float32)

        # Convert normalized metadata to float32.
        sx = np.array(sx, dtype=np.float32)
        sz = np.array(sz, dtype=np.float32)
        snapit = np.array(snapit, dtype=np.float32)

        # Stack original elastic wavefield components into a 2-channel tensor:
        #     vxz[0] = Vx
        #     vxz[1] = Vz
        #
        # This is part of the condition c in the conditional diffusion model.
        vxz = np.stack((vx, vz), axis=0)

        # Stack clean P-wave components into a 2-channel tensor:
        #     vxz_p[0] = Vx^p
        #     vxz_p[1] = Vz^p
        #
        # This is the target x_0 that the diffusion model learns to generate.
        vxz_p = np.stack((vx_p, vz_p), axis=0)

        # Stack auxiliary/intermediate P-wave components.
        mid_p = np.stack((mid_vxp, mid_vzp), axis=0)

        # Stack normalized velocity models:
        #     vel[0] = normalized vp
        #     vel[1] = normalized vs
        #
        # Together with vxz, these form the main physical conditioning variables.
        vel = np.stack((vp, vs), axis=0)

        # Stack normalized source coordinates and snapshot index.
        # This can be used as additional metadata by training or inference scripts.
        loc = np.stack((sx, sz, snapit), axis=0)

        # Empty dictionary retained for compatibility with diffusion-code APIs
        # that expect an additional dictionary of keyword conditions.
        out_dict = {}

        return vxz_p, vxz, mid_p, vel, loc, out_dict