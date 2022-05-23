# SNARE Dataset

SNARE dataset and code for MATCH and LaGOR models. 

## Paper and Citation

[Language Grounding with 3D Objects](https://arxiv.org/abs/2107.12514)

```
@article{snare,
  title={Language Grounding with {3D} Objects},
  author={Jesse Thomason and Mohit Shridhar and Yonatan Bisk and Chris Paxton and Luke Zettlemoyer},
  journal={arXiv},
  year={2021},
  url={https://arxiv.org/abs/2107.12514}
}
```

## Installation

#### Clone
```bash
$ git clone https://github.com/snaredataset/snare.git

$ virtualenv -p $(which python3) --system-site-packages snare_env # or whichever package manager you prefer
$ source snare_env/bin/activate

$ pip install --upgrade pip
$ pip install -r requirements.txt
```  
Edit `root_dir` in [cfgs/train.yaml](cfgs/train.yaml) to reflect your working directory.

#### Download Data and Checkpoints 
Download pre-extracted image features, language features, and pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1rExJT7LYJ0piZz6s54PaLOKWNElbuGrU?usp=sharing) and put them in the `data/` folder. 

## Usage

#### Zero-shot CLIP Classifier
```bash
$ python train.py train.model=zero_shot_cls train.aggregator.type=maxpool 
```

#### MATCH
```bash
$ python train.py train.model=single_cls train.aggregator.type=maxpool 
```

#### LaGOR
```bash
$ python train.py train.model=rotator train.aggregator.type=two_random_index train.lr=5e-5 train.rotator.pretrained_cls=<path_to_pretrained_single_cls_ckpt>
```

## Scripts

Run [`scripts/train_classifiers.sh`](scripts/train_classifiers.sh) and [`scripts/train_rotators.sh`](scripts/train_rotators.sh) to reproduce the results from the paper.

To train the rotators, edit [`scripts/train_rotators.sh`](scripts/train_rotators.sh) and replace the `PRETRAINED_CLS` with the path to the checkpoint you wish to use to train the rotator:
```
PRETRAINED_CLS="<root_path>/clip-single_cls-random_index/checkpoints/<ckpt_name>.ckpt'"
```

## Preprocessing

If you want to extract CLIP vision and language features from raw images:

1. Download models-screenshot.zip from [ShapeNetSem](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetSem.v0/), and extract it inside `./data/`.
2. Edit and run `python scripts/extract_clip_features.py` to save `shapenet-clipViT32-frames.json.gz` and `langfeat-512-clipViT32.json.gz` 