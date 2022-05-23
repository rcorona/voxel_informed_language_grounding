import os
from pathlib import Path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import numpy as np
import random
import pdb
import torch
import models
from torch.utils.data import DataLoader
import wandb

from data.dataset import CLIPGraspingDataset
from legoformer import LegoFormerM, LegoFormerS
from legoformer.util.utils import load_config
from legoformer.data.datamodule import ShapeNetDataModule

@hydra.main(config_path="cfgs", config_name="train")
def main(cfg):
    # set random seeds
    if cfg['train']['random_seed'] == '':
        seed = int(random.random() * 1e5)
    else:
        seed = cfg['train']['random_seed']

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path \
        if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None

    ## Need legoformer config for both tasks. 
    if cfg['train']['model'] == 'transformer':
        legoformer_class = LegoFormerM
        cfg_path = os.path.join(cfg['legoformer_paths']['cfg'], 'legoformer_m.yaml')
        net_type = 'legoformer_m'

        legoformer_cfg = load_config(cfg_path)

        # Load model and data module
        data_module = ShapeNetDataModule(legoformer_cfg.data)
    else: 
        data_module = None

    # dataset
    train = CLIPGraspingDataset(cfg, mode='train', legoformer_data_module=data_module)
    valid = CLIPGraspingDataset(cfg, mode='valid')
    test = CLIPGraspingDataset(cfg, mode='test')

    fname = '{epoch:04d}-{val_acc:.5f}'

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=checkpoint_path,
        filename=fname, 
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    # Compute number of epochs before val based on desired number of steps before. 
    val_freq = cfg['train']['val_freq']
    steps_per_epoch = int(len(train) / cfg['train']['batch_size']) + 1
    val_epochs = int(val_freq / steps_per_epoch)
    val_epochs = max(val_epochs, 1)

    trainer = Trainer(
        gpus=[0],
        fast_dev_run=cfg['debug'],
        checkpoint_callback=checkpoint_callback,
        max_epochs=cfg['train']['max_epochs'],
        check_val_every_n_epoch=val_epochs,
        **cfg.trainer # Extra arguments that come from config itself. 
    )

    # model
    model = models.names[cfg['train']['model']](cfg)

    # Preprocess VGG16 features into folder if not done so alread. 
    if train.use_imgs: 
        train.preprocess_vgg16(model)

    # resume epoch and global_steps
    if last_checkpoint and cfg['train']['load_from_last_ckpt']:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    trainer.fit(
        model,
        train_dataloader=DataLoader(train, batch_size=cfg['train']['batch_size'], num_workers=0),
        val_dataloaders=DataLoader(valid, batch_size=cfg['train']['batch_size'], num_workers=0),
    )

if __name__ == "__main__":
    main()
