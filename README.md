![LOGO](https://github.com/DeepWave-Kaust/DiffSeparation-pub/blob/main/logo/logo.jpg)
<div align="center">

<h3><strong>Physics-informed conditional diffusion model for generalizable elastic wave-mode separation</strong></h2>

<h4>Shijun Cheng, Xinru Mu, Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>


## Overview

This repository provides the official implementation of the physics-informed conditional diffusion model (PICDM) for generalizable elastic wave-mode separation.

The proposed method learns to generate the separated P-wave mode from elastic wavefield components using a conditional diffusion model. The model is conditioned on the original elastic wavefield and the corresponding P- and S-wave velocity models. A physics-informed loss and optional physics-guided sampling are incorporated to improve the physical consistency of the separated wave modes.

---

## Project structure

This repository is organized as follows:

```text
DiffSeparation-pub/
├── diff_separation/       # Python library for diffusion-based P/S wave-mode separation
├── data_generation/       # Scripts for generating training and testing datasets
├── logo/                  # Repository logo
├── environment.yml        # Conda environment file
├── install_env.sh         # Installation script
└── README.md
````

Main folders:

* :open_file_folder: **diff_separation**: Python library containing the PICDM training and sampling codes.
* :open_file_folder: **data_generation**: Python scripts for generating the training and testing datasets.
* :open_file_folder: **logo**: Logo image used in this repository.

---

## Supplementary files

To ensure reproducibility, we provide the velocity models required for dataset generation and the pretrained diffusion models through Zenodo:

**Zenodo DOI:** https://doi.org/10.5281/zenodo.20737311

Because the fully generated training and testing wavefield datasets are too large to upload directly, we provide the velocity models required to regenerate them. The executable data-generation scripts are included in the `data_generation` folder of this GitHub repository.

---

### Velocity models

The file `dataset.zip` contains the velocity models used to generate both the training and testing datasets. After unzipping `dataset.zip`, the velocity models can be found in the `velocity_model` folder.

The included files are:

```text
vp_train.npy
vs_train.npy

seamarid_vp.npy
seamarid_vs.npy
overthrust_vp.npy
overthrust_vs.npy
marmousi_small_vp.npy
marmousi_small_vs.npy
marmousi_vp.npy
marmousi_vs.npy
otway_vp.npy
otway_vs.npy
```

The files `vp_train.npy` and `vs_train.npy` are the P-wave and S-wave velocity models used to generate the training dataset.

The remaining velocity models are used to generate the testing datasets in the numerical experiments, including:

* SEAM Arid
* Overthrust
* Marmousi-small
* Marmousi
* Otway

---

### Generating the datasets

After downloading and extracting `dataset.zip`, the training and testing datasets can be regenerated using the scripts in the `data_generation` folder.

To generate the training dataset, run:

```bash
python ./data_generation/main_traindata.py
```

To generate the testing dataset, run:

```bash
python ./data_generation/main_testdata.py
```

---

### Trained models

The file `trained_model.zip` contains the pretrained diffusion models used in the paper. After unzipping the file, users will find three model checkpoints:

```text
trained_model_singlefreq.pt
trained_model_multifreq.pt
trained_model_noisydata.pt
```

The meaning of each checkpoint is as follows:

* **`trained_model_singlefreq.pt`**: the model trained on the 12 Hz single-frequency training dataset. This model is used for the main numerical experiments in the paper.
* **`trained_model_multifreq.pt`**: the model trained on the multi-frequency dataset. This model corresponds to the multi-frequency experiment discussed in the Discussion section.
* **`trained_model_noisydata.pt`**: the model trained using noisy wavefield data. This model corresponds to the noise robustness experiment discussed in the Discussion section.

---

## Getting started :space_invader: :robot:

To reproduce the results, we recommend creating the Conda environment using the provided `environment.yml` file.

Simply run:

```bash
./install_env.sh
```

The installation may take some time. If you see the word `Done!` in your terminal at the end of the installation, the environment has been successfully created.

Remember to activate the environment before running the code:

```bash
conda activate diffseparation
```

---

## Running the code :page_facing_up:

After downloading the supplementary files from Zenodo and installing the environment, you can run the training and inference scripts.

### Training

To train the PICDM model, run:

```bash
python ./diff_separation/train.py
```

By default, the training uses the full 1000-step diffusion process with `timestep_respacing=""`.

---

### Sampling / inference

To test the performance of a trained model, run:

```bash
python ./diff_separation/sample.py
```

For the accelerated DDIM sampling setting used in the paper, set:

```bash
python ./diff_separation/sample.py --use_ddim True --timestep_respacing ddim50
```

Here, `timestep_respacing="ddim50"` selects 50 timesteps from the original 1000-step diffusion process, enabling 50-step DDIM accelerated sampling.

If physics-guided sampling is used, the sampling script can be run with:

```bash
python ./diff_separation/sample.py --use_ddim True --timestep_respacing ddim50 --pde_guide True --scale_factor 5
```

---

## Reproducibility workflow

A typical workflow for reproducing the reported results is:

1. Clone this repository:

```bash
git clone https://github.com/DeepWave-KAUST/DiffSeparation-pub/
cd DiffSeparation-pub
```

2. Create and activate the Conda environment:

```bash
./install_env.sh
conda activate diffseparation
```

3. Download `dataset.zip` and `trained_model.zip` from Zenodo:

https://doi.org/10.5281/zenodo.20737311

4. Unzip `dataset.zip` and use the velocity models to generate the training and testing datasets:

```bash
python ./data_generation/main_traindata.py
python ./data_generation/main_testdata.py
```

5. Unzip `trained_model.zip` and use the provided pretrained models for inference.

6. Run the sampling script to reproduce the P-wave mode separation results.

---

## Hardware and environment

All experiments were conducted on a workstation equipped with an Intel(R) Xeon(R) CPU @ 2.10 GHz and a single NVIDIA A100 GPU.

Different hardware or software configurations may require minor adjustments. If your GPU memory is insufficient for the default training configuration, please reduce the `batch_size` argument in `diff_separation/train.py`.

---

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{cheng2025physics,
  title={Physics-informed conditional diffusion model for generalizable elastic wave-mode separation},
  author={Cheng, Shijun and Mu, Xinru and Alkhalifah, Tariq},
  journal={arXiv preprint arXiv:2506.23007},
  year={2025}
}
@inproceedings{cheng2025generative,
  title={A generative neural operator for seismic wavefield representation},
  author={Cheng, S and Taufik, MH and Alkhalifah, T},
  booktitle={86th EAGE Annual Conference \& Exhibition},
  volume={2025},
  number={1},
  pages={1--5},
  year={2025},
  organization={European Association of Geoscientists \& Engineers}
}
```

You may also cite the Zenodo record for the released velocity models and trained models:

```bibtex
@dataset{cheng2026diffseparation,
  author       = {Cheng, Shijun},
  title        = {Dataset and Trained Models for "Physics-informed
                   conditional diffusion model for generalizable
                   elastic wave-mode separation"},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20737311},
  url          = {https://doi.org/10.5281/zenodo.20737311}
}
```

---

## License

Please refer to the license file in this repository for usage terms.


