from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random

def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    dataset = BasicDataset(
        all_files,
        class_cond=class_cond,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def normalizer_vp(x, dmin=1500, dmax=5000):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_vs(x, dmin=780, dmax=3500):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_it(x, dmin=0, dmax=1500):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_coords(x, dmin=0, dmax=256):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def denormalizer_vel(x, dmin=1.5, dmax=4.5):
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


class BasicDataset(Dataset):
    def __init__(self, paths, class_cond=False):
        super().__init__()
        self.local_dataset = paths
        self.class_cond = class_cond

    def __len__(self):
        return len(self.local_dataset)

    def __getitem__(self, idx):
        path = self.local_dataset[idx]

        dict = sio.loadmat(path)
        vx = dict['vx']
        vx_p = dict['vx_p']
        vz = dict['vz']
        vz_p = dict['vz_p']
        mid_vxp = dict['mid_vxp']
        mid_vzp = dict['mid_vzp']

        vp = dict['vp']
        vs = dict['vs']
        vp = normalizer_vp(vp)
        vs = normalizer_vs(vs)

        sx = dict['sx']
        sz = dict['sz']
        snapit = dict['snapit']

        sx = normalizer_coords(sx)
        sz = normalizer_coords(sz)
        snapit = normalizer_it(snapit)

        vx = np.array(vx, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        vx_p = np.array(vx_p, dtype=np.float32)
        vz_p = np.array(vz_p, dtype=np.float32)
        mid_vxp = np.array(mid_vxp, dtype=np.float32)
        mid_vzp = np.array(mid_vzp, dtype=np.float32)

        vp = np.array(vp, dtype=np.float32)
        vs = np.array(vs, dtype=np.float32)

        sx = np.array(sx, dtype=np.float32)
        sz = np.array(sz, dtype=np.float32)
        snapit = np.array(snapit, dtype=np.float32)

        vxz = np.stack((vx, vz), axis=0)
        vxz_p = np.stack((vx_p, vz_p), axis=0)
        mid_p = np.stack((mid_vxp, mid_vzp), axis=0)
        vel = np.stack((vp, vs), axis=0)

        loc = np.stack((sx, sz, snapit), axis=0)

        out_dict = {}
        return vxz_p, vxz, mid_p, vel, loc, out_dict
