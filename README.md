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

## Oxford-IIIT Pet segmentation and image classification

`train_cifar10.py` now defaults to a real Oxford-IIIT Pet foreground/background segmentation task.
Oxford-IIIT Pet provides trimap segmentation labels; the script maps pet pixels to foreground,
background pixels to background, and ignores trimap boundary pixels during loss and metric calculation.
CIFAR-10 pseudo segmentation is still available for comparison, but Oxford-IIIT Pet is the recommended
segmentation dataset.

For both segmentation and classification, the script maps an image into a time-feature sequence:

* x axis (image width) -> time
* y axis (image height * channels) -> features

In other words:

* a `(3, 32, 32)` CIFAR-10 image becomes a sequence of length `32`, and each time step has `96` features
* a `(3, 64, 64)` Oxford-IIIT Pet image becomes a sequence of length `64`, and each time step has `192` features
* a `(3, 64, 64)` tiny-ImageNet image becomes a sequence of length `64`, and each time step has `192` features
* in segmentation mode, the Conformer output is decoded into dense `(2, height, width)` foreground/background logits

The training script now supports:

* `--task segmentation` for Oxford-IIIT Pet real segmentation (default)
* `--dataset cifar10` for CIFAR-10 pseudo segmentation
* `--task classification` for CIFAR-10, tiny-ImageNet, and ImageNet Mini classification
* `--dataset oxford-pet`, `--dataset cifar10`, `--dataset tiny-imagenet`, and `--dataset imagenet-mini`
* `--encoder-block-mode full|attention-only|convolution-only` for attention/convolution ablations
* train / validation / test split
* checkpoint saving: `last.pt` and `best.pt`
* resume training with `--resume`
* evaluation-only mode with `--eval-only`
* training history saved to `history.json`
* per-batch logging to a txt file

Run a quick Oxford-IIIT Pet segmentation smoke test:

```bash
python train_cifar10.py --dataset oxford-pet --epochs 1 --train-subset 128 --val-subset 64 --test-subset 64 --batch-size 32 --output-dir runs/oxford_pet_smoke
```

Run a fuller Oxford-IIIT Pet segmentation experiment:

```bash
python train_cifar10.py --dataset oxford-pet --epochs 50 --batch-size 128 --image-size 256 --loss ce-dice --output-dir runs/oxford_pet_segmentation_256_cedice_50ep
```

Run an attention-only Oxford-IIIT Pet smoke test:

```bash
python train_cifar10.py --dataset oxford-pet --epochs 1 --batch-size 4 --image-size 256 --loss ce-dice --encoder-block-mode attention-only --train-subset 16 --val-subset 16 --test-subset 16 --output-dir runs/oxford_pet_segmentation_attention_only_smoke
```

Export original/mask comparison images from the best checkpoint:

```bash
python export_cifar10_segmentation_examples.py --checkpoint runs/oxford_pet_segmentation_256_cedice_50ep/best.pt --output-dir runs/oxford_pet_segmentation_256_cedice_50ep/val_blue_red_comparisons --output-kind comparison --per-class 10
```

Run CIFAR-10 pseudo segmentation:

```bash
python train_cifar10.py --task segmentation --dataset cifar10 --epochs 20 --batch-size 128 --output-dir runs/cifar10_segmentation
```

Run CIFAR-10 classification:

```bash
python train_cifar10.py --task classification --dataset cifar10 --epochs 20 --batch-size 128 --output-dir runs/cifar10_conformer
```

Run tiny-ImageNet:

```bash
python train_cifar10.py --task classification --dataset tiny-imagenet --data-dir /path/to/tiny-imagenet-200 --epochs 20 --batch-size 1024 --lr 0.001 --output-dir runs/tiny_imagenet
```

Download ImageNet Mini from Kaggle:

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('ifigotin/imagenetmini-1000', output_dir='data/imagenetmini-1000')"
```

Run ImageNet Mini:

```bash
python train_cifar10.py --task classification --dataset imagenet-mini --data-dir data/imagenetmini-1000 --epochs 20 --batch-size 128 --lr 0.001 --image-size 64 --output-dir runs/imagenet_mini
```

Resume training:

```bash
python train_cifar10.py --task segmentation --dataset oxford-pet --epochs 100 --batch-size 128 --lr 0.001 --resume runs/oxford_pet_segmentation/last.pt --output-dir runs/oxford_pet_segmentation
```

Evaluate the best checkpoint:

```bash
python train_cifar10.py --eval-only --dataset oxford-pet --resume runs/oxford_pet_segmentation/best.pt
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
