![LOGO](https://github.com/DeepWave-Kaust/DiffSeparation-pub/blob/main/logo/logo.jpg)

<div align="center">

<h3><strong>Physics-informed conditional diffusion model for generalizable elastic wave-mode separation</strong></h2>

<h4>Shijun Cheng, Xinru Mu, Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>

# Project structure
This repository is organized as follows:

* :open_file_folder: **diff_separation**: python library containing diffusion P/S separation.
* :open_file_folder: **data_generation**: python library to generate the training and testing dataset.

## Supplementary files
To ensure reproducibility, we provide the the data set for both training and sampling stages and our trainined GNO model. 

* **Training and Testing data set**:
Since the dataset is so large, we provide the velocity models [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0091/dataset.zip?csf=1&web=1&e=kUj6Wl) used for training and testing. You can use the data generation script and the velocity models to generate training and testing datasets.

For genereting the training dataset, you can directly run:
```
python ./data_generation/main_traindata.py
```

For genereting the testing dataset, you can directly run:
```
python ./data_generation/main_testdata.py
```

* **Trained model**:
Download our trained model [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0091/trainmodel.zip?csf=1&web=1&e=rWsOYg). 


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate diffseparation
```

## Running code :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can run the training and inference code. 

For traning, you can directly run:
```
python ./diff_separation/train.py
```

When you test the performance of our trained model, you can use the testing data we provide, and directly run:
```
python ./diff_separation/sample.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `diff_separation/train.py` file.

## Cite us 
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

