#
# Adapted from code developed by Farid Yagubbayli <faridyagubbayli@gmail.com> | <farid.yagubbayli@tum.de>
#

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from legoformer.data.datamodule import ShapeNetDataModule
from legoformer.util.utils import load_config
from legoformer import LegoFormerM, LegoFormerS
from pytorch_lightning import Trainer


model_zoo = {
    'legoformer_m': LegoFormerM,
    'legoformer_s': LegoFormerS,
}


if __name__ == '__main__':
    # Get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config file", type=str)
    parser.add_argument("--ckpt_path", help="Model checkpoint path", type=str, default=None)
    parser.add_argument("--views", help="Number of views", type=int, default=8)
    parser.add_argument("--task", choices=['train', 'test'], help='Train or test', default='test')

    args = parser.parse_args()

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    n_views = args.views

    # Load config file
    cfg = load_config(config_path)

    # Enforce some config parameters
    cfg.trainer.precision = 32
    cfg.data.constants.n_views = n_views
    cfg.optimization.metrics = ['iou']

    if cfg.seed != -1:
        pl.seed_everything(cfg.seed)

    net_type = cfg.network.type
    print('Network type:', net_type, ' n_views:', n_views)

    # Load model and data module
    model = model_zoo[cfg.network.type]

    if args.ckpt_path: 
        model = model.load_from_checkpoint(ckpt_path, config=cfg)
    else:
        model = model(cfg)

    data_module = ShapeNetDataModule(cfg.data)

    checkpoint_callback = ModelCheckpoint(
        monitor='metrics/val_iou',
        mode='max',
        save_top_k=1,
        dirpath='./checkpoints',
        save_last=True,
    )

    # Start evaluation process
    trainer = Trainer(
        logger=False,
        checkpoint_callback=checkpoint_callback,
        #accelerator='ddp',
        **cfg.trainer,
    )

    if args.task == 'train':
        # Fit model. 
        trainer.fit(
            model,
            train_dataloader=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

    elif args.task == 'test': 
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
