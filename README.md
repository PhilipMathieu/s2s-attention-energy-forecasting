# Energy Forecasting using S2S, S2S with attention, Regular RNN, and DNN models, as applied to run-of-river electrical production

This project adapts code from Ljubisa Sehovac. For the original work, please see the [upstream GitHub repository](https://github.com/sehovaclj/Thesis-work_energy-forecasting) or [L. Sehovac and K. Grolinger, "Deep Learning for Load Forecasting: Sequence to Sequence Recurrent Neural Networks With Attention," in _IEEE Access_, vol. 8, pp. 36411-36426, 2020, doi: 10.1109/ACCESS.2020.2975738](https://doi.org/10.1109/ACCESS.2020.2975738). [The original README is located here.](./readme_upstream.md)

## Installation

Install dependencies:
```
conda install numpy pandas matplotlib
```

Install PyTorch following [platform-specific directions](https://pytorch.org/get-started/locally/) or use the following command if you are on Linux or Windows and do not have CUDA:

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Run

```
python -B all_S2S_models.py
```