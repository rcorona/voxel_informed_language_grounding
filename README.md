# VLG 
Code for the paper: 

[**Voxel-informed Language Grounding**](http://arxiv.org/abs/2205.09710)\
Rodolfo Corona, Shizhan Zhu, Dan Klein, Trevor Darrell\
http://arxiv.org/abs/2205.09710 \
ACL 2022 

Natural language applied to natural 2D images describes a fundamentally 3D world. 
We present the Voxel-informed Language Grounder (VLG), a language grounding model that leverages _3D geometric information_ in the form of voxel maps derived from the visual input using a volumetric reconstruction model.  
We show that VLG significantly improves grounding accuracy on [SNARE](https://arxiv.org/abs/2107.12514), an object reference game task.
At the time of writing, VLG holds the top place on the SNARE [leaderboard](https://github.com/snaredataset/snare\#leaderboard) achieving SOTA results with a 2.0% absolute improvement.

# Cite
```
@InProceedings{Corona-Zhu-Klein-Darrell:2022:VLG,
  title     = {Voxel-informed Language Grounding},
  author    = {Rodolfo Corona and Shizhan Zhu and Dan Klein and Trevor Darrell},
  booktitle = {Proceedings of ACL},
  address   = {},
  pages     = {},
  month     = {May},
  year      = {2022},
}
```

# References

This codebase was built through modifications to the [SNARE](https://github.com/snaredataset/snare) (Thomason et al. 2021) and [LegoFormer](https://github.com/faridyagubbayli/LegoFormer) (Yagubbayli et al. 2021) codebases. 

# Download or Pre-train LegoFormer Weights

You may either download our pre-trained LegoFormer weights or train your own. 

In either case, you will want to create a `checkpoints` folder: 

```
cd snare-master
mkdir checkpoints
```

## Download

You may download the weights from this [link](https://drive.google.com/file/d/1FQuZDJOwSvqzzPONNsho8mvCgbFdjRRH/view?usp=sharing) and place them in the `checkpoints` folder you created above. 

The configuration file paths should already point to the correct file if set up correctly, in which case no further steps are needed to set up LegoFormer. 

## Pre-train

**Note:** You will need a little over 25GB of GPU memory to pre-train your own LegoFormer weights. 

If pre-training, you will need to create a conda environment specifically for LegoFormer: 

```
conda create -n legoformer python=3.7
conda activate legoformer
bash install_legoformer_dependencies.sh 
```

Download and unpack the ShapeNet data required for pre-training into the data folder. 

```
mkdir data
cd data

wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz 

tar -zxvf ShapeNetRendering.tgz
tar -zxvf ShapeNetVox32.tgz

cd ..
```

Train a LegoFormer model, it will be stored in the `checkpoints` folder. 

```
python train_legoformer.py legoformer/config/legoformer_m.yaml --views 8 --task train
```

The configuration file is currently set for 8 epochs, which will train the model for approximately 86K steps. 

After this, both the last and the best validation performing LegoFormer checkpoints should be stored in the `checkpoints` directory. 

For each experiment configuration file in `cfgs/` (see **Update Configuration Files** below) you will need to modify the `legoformer_m` path under `legoformer_paths` to point to the checkpoint you would like to use (we recommend the best performing checkpoint, but in practice we found the last checkpoint to work just as well). 

# Run VLG Experiments

## Setup 

### Installation 

Deactivate the LegoFormer environment and create the SNARE environment. 

```
conda deactivate legoformer

conda create -n snare_env python=3.6
conda activate snare_env
pip install -r requirements.txt 
```

Download ShapeNetSem images and extract CLIP features using the script from the SNARE repo. 

```
cd data
wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetSem.v0/models-screenshots.zip
unzip models-screenshots.zip
rm models-screenshots.zip

cd ..
python scripts/extract_clip_features.py
gzip -d shapenet-clipViT32-frames.json.gz
gzip -d langfeat-512-clipViT32.json.gz
```

### Update Configuration Files

Each experiment is specified by a configuration file in the `cfgs` folder, e.g. `vlg.yaml`. 
For each one you'd like to use, you will need to modify the `root_dir` variable to point to your installation of this repo. 

Additionally, if you would like to use custom LegoFormer weights which you trained, you will need to change the `legoformer_m` variable to point to that checkpoint's path (the current path points to the default weights provided with this repo). 

You may find the lines to update by searcing for `TODO` statements in the configuration files. 

## Run VLG

Running the following script will train the VLG model using the configuration from the paper under a random seed, where `EXP_NAME` is the name to set for the directory where the results will be stored (set this name as desired, e.g. "VLG"). 
The script will display the best performance on the validation set after each epoch. 

**Note:** The first time the script is run, VGG 16 features will be pre-extracted in order to speed up computation moving forward. This step may take a few hours to complete. 

```
bash scripts/train.sh EXP_NAME vlg.yaml
``` 

The best performing model checkpoint will be stored in the following path: 

```
snap/EXP_NAME/checkpoints/'epoch=00XX-val_acc=0.YYYYY.ckpt'
```

## Run Ablation Experiments

The ablation experiments are run similarly. 

### Run MLP Ablation

```
bash scripts/train.sh EXP_NAME vlg_mlp_abl.yaml
```

### Run CLIP Ablation

```
bash scripts/train.sh EXP_NAME vlg_clip_abl.yaml
```

### Run VGG Ablation

```
bash scripts/train.sh EXP_NAME vlg_vgg_abl.yaml
```

# Questions

Please reach out with any questions either by raising an issue here on this Github repo or by emailing Rodolfo Corona (email address in paper). 

Thanks!
