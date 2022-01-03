# RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts.

by [Xudong Wang](http://people.eecs.berkeley.edu/~xdwang/), [Long Lian](https://github.com/TonyLianLong/), [Zhongqi Miao](https://scholar.google.com/citations?user=at4m2mYAAAAJ&hl=en), [Ziwei Liu](https://liuziwei7.github.io/) and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley, ICSI and NTU

<em>International Conference on Learning Representations (ICLR), 2021. **Spotlight Presentation**</em>

[Project Page](http://people.eecs.berkeley.edu/~xdwang/projects/RIDE/) | [PDF](http://people.eecs.berkeley.edu/~xdwang/papers/ICLR2021_RIDE.pdf) | 
[Preprint](https://arxiv.org/abs/2010.01809) | [OpenReview](https://openreview.net/forum?id=D9I3drBz4UC) | [Slides](http://people.eecs.berkeley.edu/~xdwang/projects/RIDE/ICLR2021-RIDE-10mins-V4.pdf ) | [Citation](#citation)

<img src="title-img.png" width="100%" />

This repository contains an official re-implementation of RIDE from the authors, while also has plans to support other works on long-tailed recognition. Further information please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu) and [Long Lian](mailto:longlian@berkeley.edu).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wang2021longtailed,
  title={Long-tailed Recognition by Routing Diverse Distribution-Aware Experts},
  author={Xudong Wang and Long Lian and Zhongqi Miao and Ziwei Liu and Stella Yu},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=D9I3drBz4UC}
}
```

## Supported Methods for Long-tailed Recognition:
- [x] RIDE
- [x] Cross-Entropy (CE) Loss
- [x] Focal Loss
- [x] LDAM Loss
- [x] Decouple: cRT (limited support for now)
- [x] Decouple: tau-normalization (limited support for now)

## Updates
[04/2021] Pre-trained models are avaliable in model zoo.  

[12/2020] We added an approximate GFLops counter. See usages below. We also refactored the code and fixed a few errors.  

[12/2020] We have limited support on cRT and tau-norm in `load_stage1` option and `t-normalization.py`, please look at the code comments for instructions while we are still working on it.

[12/2020] Initial Commit. We re-implemented RIDE in this repo. LDAM/Focal/Cross-Entropy loss is also re-implemented (instruction below).

## Table of contents
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=1 orderedList=false} -->

<!-- code_chunk_output -->

- [RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](#ride-long-tailed-recognition-by-routing-diverse-distribution-aware-experts)
  - [Supported Methods for Long-tailed Recognition:](#supported-methods-for-long-tailed-recognition)
  - [Updates](#updates)
  - [Table of contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Dataset Preparation](#dataset-preparation)
  - [How to get pretrained checkpoints](#how-to-get-pretrained-checkpoints)
  - [Training and Evaluation Instructions](#training-and-evaluation-instructions)
  - [FAQ](#faq)
  - [How to get support from us?](#how-to-get-support-from-us)
  - [Pytorch template](#pytorch-template)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* tqdm (Used in `test.py`)
* tensorboard >= 1.14 (for visualization)
* pandas
* numpy

### Hardware requirements
8 GPUs with >= 11G GPU RAM are recommended. Otherwise the model with more experts may not fit in, especially on datasets with more classes (the FC layers will be large). We do not support CPU training, but CPU inference could be supported by slight modification.

## Dataset Preparation
CIFAR code will download data automatically with the dataloader. We use data the same way as [classifier-balancing](https://github.com/facebookresearch/classifier-balancing). For ImageNet-LT and iNaturalist, please prepare data in the `data` directory. ImageNet-LT can be found at [this link](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf). iNaturalist data should be the 2018 version from [this](https://github.com/visipedia/inat_comp) repo (Note that it requires you to pay to download now). The annotation can be found at [here](https://github.com/facebookresearch/classifier-balancing/tree/master/data). Please put them in the same location as below:

```
data
├── cifar-100-python
│   ├── file.txt~
│   ├── meta
│   ├── test
│   └── train
├── cifar-100-python.tar.gz
├── ImageNet_LT
│   ├── ImageNet_LT_open.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_val.txt
│   ├── test
│   ├── train
│   └── val
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_val.txt
    └── train_val2018
```

## How to get pretrained checkpoints
We have a [model zoo](MODEL_ZOO.md) available.

## Training and Evaluation Instructions
### Imbalanced CIFAR 100/CIFAR100-LT
##### RIDE Without Distill (Stage 1)
```
python train.py -c "configs/config_imbalance_cifar100_ride.json" --reduce_dimension 1 --num_experts 3
```
Note: `--reduce_dimension 1` means set reduce dimension to True. The template has an issue with bool arguments so int argument is used here. However, any non-zero value will be equivalent to bool True.

##### RIDE With Distill (Stage 1)
```
python train.py -c "configs/config_imbalance_cifar100_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint path_to_checkpoint
```

Distillation is not required but could be performed if you'd like further improvements.

##### RIDE Expert Assignment Module Training (Stage 2)
```
python train.py -c "configs/config_imbalance_cifar100_ride_ea.json" -r path_to_stage1_checkpoint --reduce_dimension 1 --num_experts 3
```

Note: different runs will result in different EA modules with different trade-off. Some modules give higher accuracy but require higher FLOps. Although the only difference is not underlying ability to classify but the "easiness to satisfy and stop". You can tune the `pos_weight` if you think the EA module consumes too much compute power or is using too few expert.

<!--
##### cRT (load from a checkpoint without linear and freezes the pretrained parameters)
This part is not finalized and will probably change.
```
python train.py --load_crt path_to_cRT_checkpoint -c path_to_config --reduce_dimension 1 --num_experts 3
```

##### t-norm
This part is not finalized and will probably change.
Please see `t-normalization.py` for usages. It requires a hyperparemeter from the decouple paper.
-->

### ImageNet-LT
#### RIDE Without Distill (Stage 1)
##### ResNet 10
```
python train.py -c "configs/config_imagenet_lt_resnet10_ride.json" --reduce_dimension 1 --num_experts 3
```

##### ResNet 50
```
python train.py -c "configs/config_imagenet_lt_resnet50_ride.json" --reduce_dimension 1 --num_experts 3
```

##### ResNeXt 50
```
python train.py -c "configs/config_imagenet_lt_resnext50_ride.json" --reduce_dimension 1 --num_experts 3
```

#### RIDE With Distill (Stage 1)
##### ResNet 10
```
python train.py -c "configs/config_imagenet_lt_resnet10_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint path_to_checkpoint
```

##### ResNet 50
```
python train.py -c "configs/config_imagenet_lt_resnet50_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint path_to_checkpoint
```

##### ResNeXt 50
```
python train.py -c "configs/config_imagenet_lt_resnext50_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint path_to_checkpoint
```

#### RIDE Expert Assignment Module Training (Stage 2)
##### ResNet 10
```
python train.py -c "configs/config_imagenet_lt_resnet10_ride_ea.json" -r path_to_stage1_checkpoint --reduce_dimension 1 --num_experts 3
```

##### ResNet 50
```
python train.py -c "configs/config_imagenet_lt_resnet50_ride_ea.json" -r path_to_stage1_checkpoint --reduce_dimension 1 --num_experts 3
```

##### ResNeXt 50
```
python train.py -c "configs/config_imagenet_lt_resnext50_ride_ea.json" -r path_to_stage1_checkpoint --reduce_dimension 1 --num_experts 3
```

### iNaturalist
#### RIDE Without Distill (Stage 1)
```
python train.py -c "configs/config_iNaturalist_resnet50_ride.json" --reduce_dimension 1 --num_experts 3
```

#### RIDE With Distill (Stage 1)
```
python train.py -c "configs/config_iNaturalist_resnet50_distill_ride.json" --reduce_dimension 1 --num_experts 3 --distill_checkpoint path_to_checkpoint
```

#### RIDE Expert Assignment Module Training (Stage 2)
```
python train.py -c "configs/config_iNaturalist_resnet50_ride_ea.json" -r path_to_stage1_checkpoint --reduce_dimension 1 --num_experts 3
```

### Using Other Methods with RIDE
<!-- * LDAM: switch the config to the corresponding config -->
* Focal Loss: switch the loss to Focal Loss
* Cross Entropy: switch the loss to Cross Entropy Loss

### Test
To test a checkpoint, please put it with the corresponding config file.
```
python test.py -r path_to_checkpoint
```

Please see [the pytorch template that we use](https://github.com/victoresque/pytorch-template) for additional more general usages of this project (e.g. loading from a checkpoint, etc.).

### GFLops calculation
We provide an experimental support for approximate GFLops calculation. Please open an issue if you encounter any problem or meet inconsistency in GFLops.

You need to install `thop` package first. Then, according to your model, run `python -m utils.gflops (args)` in the project directory.

#### Examples and explanations
Use `python -m utils.gflops` to see the documents as well as explanations for this calculator.

##### ImageNet-LT
```
python -m utils.gflops ResNeXt50Model 0 --num_experts 3 --reduce_dim True --use_norm False
```
To change model, switch `ResNeXt50Model` to the ones used in your config. `use_norm` comes with LDAM-based methods (including RIDE). `reduce_dim` is used in default RIDE models. The `0` in the command line indicates the dataset.

All supported datasets:

* 0: ImageNet-LT
* 1: iNaturalist
* 2: Imbalance CIFAR 100

##### iNaturalist
```
python -m utils.gflops ResNet50Model 1 --num_experts 3 --reduce_dim True --use_norm True
```

##### Imbalance CIFAR 100
```
python -m utils.gflops ResNet32Model 2 --num_experts 3 --reduce_dim True --use_norm True
```

##### Special circumstances: calculate the approximate GFLops in models with expert assignment module
We provide a `ea_percentage` for specifying the percentage of data that pass each expert. Note that you need to switch to the `EA` model as well since you actually use `EA` model instead of the original model in training and inference.

An example:

```
python -m utils.gflops ResNet32EAModel 2 --num_experts 3 --reduce_dim True --use_norm True --ea_percentage 40.99,9.47,49.54
```

## FAQ
See [FAQ](FAQ.md).

## How to get support from us?
If you have any general questions, feel free to email us at `longlian at berkeley.edu` and `xdwang at eecs.berkeley.edu`. If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 

## Pytorch template
This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). The readme of the template explains its functionality, although we try to list most frequently used ones in this readme.

### License
This project is licensed under the MIT License. See [LICENSE](https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). The pytorch template is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)

The ResNet and ResNeXt in `fb_resnets` are based on from [Classifier-Balancing/Decouple](https://github.com/facebookresearch/classifier-balancing). The ResNet in `ldam_drw_resnets`/LDAM loss/CIFAR-LT are based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW). KD implementation takes references from CRD/[RepDistiller](https://github.com/HobbitLong/RepDistiller).
