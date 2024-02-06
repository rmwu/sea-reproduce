# Sample, estimate, aggregate: A recipe for causal discovery foundation models

Official implementation of SEA (under review).

## Overview

Inspired by foundation models, we propose a causal
discovery framework where a deep learning model is pretrained to resolve
predictions from classical discovery algorithms run over smaller subsets of
variables. This method is enabled by the observations that the outputs from
classical algorithms are fast to compute for small problems, informative of
(marginal) data structure, and their structure outputs as objects remain
comparable across datasets.

If you find our work interesting, please check out our paper to learn more:
[Sample, estimate, aggregate: A recipe for causal discovery foundation
models](http://arxiv.org/abs/2402.01929).

```
@article{wu2024sea,
  title={Sample, estimate, aggregate: A recipe for causal discovery
  foundation models},
  author={Wu, Menghua and Bao, Yujia and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv 2402.01929},
  year={2024}
}
```

## Installation

```
conda create -y --name sea pip python=3.10
conda activate sea

pip install tqdm pyyaml numpy=1.23 pandas matplotlib seaborn
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning==1.9.0 torchmetrics==1.3.0 causal-learn==0.1.3.7 pulp==2.8.0 wandb
```

SEA was tested using Python 3.10 with PyTorch 1.13-cu116.
We ran training across two A6000 GPUs and inference on one V100 GPU.

## Quickstart

To run inference using our pretrained models, please modify the data and model paths in
`src/inference.sh`, specify the appropriate config file, and run:
```
./src/inference.sh
```
When benchmarking runtimes, it is assumed that `batch_size=1`.
If you do not need runtimes, you may increase `batch_size` for faster
completion.

To train your own SEA, please modify the data and model paths in
`src/train.sh`, specify the appropriate config file, change the wandb
project to your own, and run:
```
./src/train.sh
```
We recommend at least 10-20 data workers per GPU and a batch size of at least
16.

## Models

We provide pretrained weights for 3 versions of SEA under `checkpoints`:
- GIES on synthetic data with (primarily) additive noise
- FCI on synthetic data with (primarily) additive noise
- FCI on SERGIO-simulated data

Please note that, while these models are expected to work well across a
variety of synthetic datasets, there are cases like e.g. sigmoid with
multiplicative noise which are hard, given our limited training data.

## Datasets

You may download our datasets [here](https://zenodo.org/records/10611036).
Sample data split files are specified in `data`.
- Synthetic testing datasets (Erdos-Renyi and scale-free)
- SERGIO testing datasets

Our datasets follow the [DCDI](https://github.com/slachapelle/dcdi) data format.
- Each file has a suffix of `dataset_id`, which distinguishes between datasets
  generated under the same setting.
- `DAG[id].npy` is a NumPy array containing the `N*N` ground truth graph.
- `data[id].npy` is a NumPy array containing `M*N` observational data.
- `data_interv[id].npy` is a NumPy array containing `M*N` interventional data.
- `regimes[id].csv` is a length `M` text file in which every line `i` specifies
  the regime index of sample `i`.
- `interventions[id].csv` is a length `M` text file in which every line `i` specifies
  the nodes intervened in sample `i`, delimited by `,`.

## Results

Our outputs take the form of a pickled `dict`. Example parsing code is provided
in `examples/SEA-results.ipynb`.

We have uploaded the predictions of all traditional baselines and our models
to the [Zenodo archive](https://zenodo.org/records/10611036) as well.

