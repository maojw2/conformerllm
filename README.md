<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/105607164-aa878e00-5de0-11eb-8474-a12dd6ac919b.png" height=100>
  

<div align="center">

**PyTorch implementation of Conformer: Convolution-augmented Transformer for Speech Recognition.**

  
</div>

***

<p align="center"> 
     <a href="https://github.com/sooftware/jasper/blob/main/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
     <a href="https://doi.org/10.5281/zenodo.18154427">
          <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18154427.svg" alt="DOI">
     </a>
</p>

  
Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. Conformer combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence in a parameter-efficient way. Conformer significantly outperforms the previous Transformer and CNN based models achieving state-of-the-art accuracies.   

<img src="https://user-images.githubusercontent.com/42150335/105602364-aeafad80-5dd8-11eb-8886-b75e2d9d31f4.png" height=600>
  
This repository contains only model code, but you can train with conformer at [openspeech](https://github.com/openspeech-team/openspeech)
  
## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.  
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
  
```
pip install -e .
```

## Usage

```python
import torch
import torch.nn as nn
from conformer import Conformer

batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

criterion = nn.CTCLoss().to(device)

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_classes=10, 
                  input_dim=dim, 
                  encoder_dim=32, 
                  num_encoder_layers=3).to(device)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Calculate CTC Loss
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
```

### Using image-like feature maps

If your training data is stored like an image whose x-axis is time and y-axis is features, you can pass it directly
to the model as `(batch, channels, features, time)`.

```python
import torch
from conformer import Conformer

batch_size, channels, features, time = 3, 1, 80, 1024

model = Conformer(
    num_classes=10,
    input_dim=channels * features,
    encoder_dim=32,
    num_encoder_layers=3,
    input_layout="bcft",
)

inputs = torch.rand(batch_size, channels, features, time)
input_lengths = torch.LongTensor([1024, 1000, 980])
outputs, output_lengths = model(inputs, input_lengths)
```

Supported layouts:

* `btc`: `(batch, time, features)` (original behavior)
* `bft`: `(batch, features, time)`
* `bcft`: `(batch, channels, features, time)`
* `auto`: infer a supported layout when possible

For `bcft`, set `input_dim = channels * features`.

## Image classification

To test the Conformer structure on image classification, `train_cifar10.py` maps an image into a
time-feature sequence:

* x axis (image width) -> time
* y axis (image height * channels) -> features

In other words:

* a `(3, 32, 32)` CIFAR-10 image becomes a sequence of length `32`, and each time step has `96` features
* a `(3, 64, 64)` tiny-ImageNet image becomes a sequence of length `64`, and each time step has `192` features

The training script now supports:

* `--dataset cifar10` and `--dataset tiny-imagenet`
* train / validation / test split
* checkpoint saving: `last.pt` and `best.pt`
* resume training with `--resume`
* evaluation-only mode with `--eval-only`
* training history saved to `history.json`
* per-batch logging to a txt file

Run a quick smoke test:

```bash
python train_cifar10.py --epochs 1 --train-subset 1024 --val-subset 256 --test-subset 256 --batch-size 64
```

Run a fuller CIFAR-10 experiment:

```bash
python train_cifar10.py --dataset cifar10 --epochs 20 --batch-size 128 --output-dir runs/cifar10_conformer
```

Run tiny-ImageNet:

```bash
python train_cifar10.py --dataset tiny-imagenet --data-dir /path/to/tiny-imagenet-200 --epochs 20 --batch-size 1024 --lr 0.001 --output-dir runs/tiny_imagenet
```

Resume training:

```bash
python train_cifar10.py --dataset tiny-imagenet --data-dir /path/to/tiny-imagenet-200 --epochs 100 --batch-size 1024 --lr 0.001 --resume runs/tiny_imagenet/last.pt --output-dir runs/tiny_imagenet
```

Evaluate the best checkpoint:

```bash
python train_cifar10.py --eval-only --resume runs/cifar10_conformer/best.pt
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/conformer/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100.pdf)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)
- [espnet/espnet](https://github.com/espnet/espnet)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
