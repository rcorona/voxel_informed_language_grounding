import numpy as np
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import wandb

import models.aggregator as agg


class SingleClassifier(LightningModule):

    def __init__(self, cfg):
        self.optimizer = None
        super().__init__()

        self.cfg = cfg
        self.dropout = self.cfg['train']['dropout']

        # input dimensions
        self.feats_backbone = self.cfg['train']['feats_backbone']
        self.img_feat_dim = 512
        self.lang_feat_dim = 512
        self.num_views = self.cfg['data']['n_views']

        # choose aggregation method
        agg_cfg = dict(self.cfg['train']['aggregator'])
        agg_cfg['input_dim'] = self.img_feat_dim
        self.aggregator_type = self.cfg['train']['aggregator']['type']
        self.aggregator = agg.names[self.aggregator_type](agg_cfg)

        # build network
        self.build_model()

        # val progress
        self.best_val_acc = -1.0
        self.best_val_res = None

        # test progress
        self.best_test_acc = -1.0
        self.best_test_res = None

        # results save path
        self.save_path = Path(os.getcwd())

        # log with wandb
        self.log_data = self.cfg['train']['log']
        if self.log_data:
            self.run = wandb.init(
                project=self.cfg['wandb']['logger']['project'],
                config=self.cfg['train'],
                settings=wandb.Settings(show_emoji=False),
                reinit=True
            )
            wandb.run.name = self.cfg['wandb']['logger']['run_name']

    def build_model(self):
        # image encoder
        self.img_fc = nn.Sequential(
            nn.Identity()
        )

        # language encoder
        self.lang_fc = nn.Sequential(
            nn.Identity()
        )

        # finetuning layers for classification
        self.cls_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg['train']['lr'])
        return self.optimizer

    def smoothed_cross_entropy(self, pred, target, alpha=0.1):
        # From ShapeGlot (Achlioptas et. al)
        # https://github.com/optas/shapeglot/blob/master/shapeglot/models/neural_utils.py
        n_class = pred.size(1)
        one_hot = target
        one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)

    def _criterion(self, out):
        probs = out['probs']
        labels = out['labels']

        loss = self.smoothed_cross_entropy(probs, labels)

        return {
            'loss': loss
        }

    def forward(self, batch):

        # Unpack features.  
        (img1_n_feats, img2_n_feats) = batch['img_feats'] if 'img_feats' in batch else None        
        lang_feats = batch['lang_feats']
        ans = batch['ans']
        (key1, key2) =  batch['keys']
        annotation = batch['annotation']
        is_visual = batch['is_visual']

        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        # aggregate
        img1_feats = self.aggregator(img1_n_feats)
        img2_feats = self.aggregator(img2_n_feats)

        # lang encoding
        lang_enc = self.lang_fc(lang_feats)

        # normalize
        if self.cfg['train']['normalize_feats']:
            img1_feats = img1_feats / img1_feats.norm(dim=-1, keepdim=True)
            img2_feats = img2_feats / img2_feats.norm(dim=-1, keepdim=True)
            lang_enc = lang_enc / lang_enc.norm(dim=-1, keepdim=True)

        # img1 prob
        img1_enc = self.img_fc(img1_feats)
        img1_prob = self.cls_fc(torch.cat([img1_enc, lang_enc], dim=-1))

        # img2 prob
        img2_enc = self.img_fc(img2_feats)
        img2_prob = self.cls_fc(torch.cat([img2_enc, lang_enc], dim=-1))

        # cat probs
        probs = torch.cat([img1_prob, img2_prob], dim=-1)

        # num steps taken (8 for all views)
        bs = lang_enc.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_enc.device)
        if self.aggregator_type in ['maxpool', 'mean', 'gru']:
            num_steps = num_steps * 8
        elif self.aggregator_type in ['two_random_index']:
            num_steps = num_steps * 2

        test_mode = (ans[0] == -1)
        if not test_mode:
            # one-hot labels of answers
            labels = F.one_hot(ans)

            return {
                'probs': probs,
                'labels': labels,
                'is_visual': is_visual,
                'num_steps': num_steps,
            }
        else:
            return {
                'probs': probs,
                'num_steps': num_steps,
            }

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        # classifier loss
        losses = self._criterion(out)

        if self.log_data:
            wandb.log({
                'tr/loss': losses['loss'],
            })

        return dict(
            loss=losses['loss']
        )

    def validation_step(self, batch, batch_idx):
        
        # Only do a single pass (additional views have been moved to data-loading step through Dataset). 
        out = self.forward(batch)
        losses = self._criterion(out)

        probs = out['probs']
        labels = out['labels']
        visual = out['is_visual']
        num_steps = out['num_steps']

        probs = F.softmax(probs, dim=-1)
        metrics = self.compute_metrics(labels, losses, probs, visual, num_steps, out)

        return dict(
            val_loss=metrics['val_loss'],
            val_acc=metrics['val_acc'],
            metrics=metrics
        )

    def check_correct(self, labels, probs):
        guess = probs.argmax(dim=1)
        labels = labels.argmax(dim=1)
        correct = torch.eq(labels, guess).float()
        
        return correct

    def compute_metrics(self, labels, losses, probs, visual, num_steps, out):
        val_total = probs.shape[0]
        
        pred_voxels = out['reconstructions'] if 'reconstructions' in out else None
        gt_voxels = out['gt_voxels'] if 'gt_voxels' in out else None
        vmasks = out['voxel_masks'] if 'voxel_masks' in out else None
        
        # TODO change naming scheme to accomodate for test set as well. 

        # Compute correct by index in batch. 
        correct = self.check_correct(labels, probs)
        val_correct = correct.sum().item()

        # See which visual examples are correct and which aren't. 
        visual_total = visual.float().sum().item()
        visual_correct = (visual.view(-1).float() * correct).sum().item()

        nonvis_total = float(val_total) - visual_total
        nonvis_correct = val_correct - visual_correct

        val_acc = float(val_correct) / val_total
        val_visual_acc = float(visual_correct) / visual_total
        val_nonvis_acc = float(nonvis_correct) / nonvis_total

        return_dict = dict(
            val_acc=val_acc,
            val_correct=val_correct,
            val_total=val_total,
            val_visual_acc=val_visual_acc,
            val_visual_correct=visual_correct,
            val_visual_total=visual_total,
            val_nonvis_acc=val_nonvis_acc,
            val_nonvis_correct=nonvis_correct,
            val_nonvis_total=nonvis_total
        )

        for loss in losses.keys(): 
            return_dict['val_{}'.format(loss)] = losses[loss]

        return return_dict

    def validation_epoch_end(self, all_outputs, mode='vl'):
        sanity_check = True

        res = {
            'val_loss': 0.0,

            'val_correct': 0,
            'val_total': 0,

            'val_visual_correct': 0,
            'val_visual_total': 0,

            'val_nonvis_correct': 0,
            'val_nonvis_total': 0,

            'val_iou': 0
        }

        for output in all_outputs:
            metrics = output['metrics']
            res['val_loss'] += metrics['val_loss'].item()
            res['val_correct'] += metrics['val_correct']
            res['val_total'] += metrics['val_total']
            
            if res['val_total'] > 128:
                sanity_check = False

            res['val_visual_correct'] += metrics['val_visual_correct']
            res['val_visual_total'] += metrics['val_visual_total']

            res['val_nonvis_correct'] += metrics['val_nonvis_correct']
            res['val_nonvis_total'] += metrics['val_nonvis_total']

            if 'iou' in metrics: 
                res['val_iou'] += metrics['iou']

        res['val_loss'] = float(res['val_loss']) / len(all_outputs)
        res['val_acc'] = float(res['val_correct']) / res['val_total']
        res['val_visual_acc'] = float(res['val_visual_correct']) / res['val_visual_total']
        res['val_nonvis_acc'] = float(res['val_nonvis_correct']) / res['val_nonvis_total']

        # Compute IoU metric. # TODO correct for masking out invalid voxelmaps. (333/7881 in dataset).  
        res['val_iou'] = float(res['val_iou']) / (res['val_total'] * 2) # Have two objects per example.

        res = {
            f'{mode}/loss': res['val_loss'],
            f'{mode}/acc': res['val_acc'],
            f'{mode}/acc_visual': res['val_visual_acc'],
            f'{mode}/acc_nonvis': res['val_nonvis_acc'],
            f'{mode}/iou': res['val_iou']
        }

        if not sanity_check:  # only check best conditions and dump data if this isn't a sanity check

            # test (ran once at the end of training)
            if mode == 'test':
                self.best_test_res = dict(res)

            # val (keep track of best results)
            else:
                if res[f'{mode}/acc'] > self.best_val_acc:
                    self.best_val_acc = res[f'{mode}/acc']
                    self.best_val_res = dict(res)

            # results to save
            results_dict = self.best_test_res if mode == 'test' else self.best_val_res

            best_loss = results_dict[f'{mode}/loss']
            best_acc = results_dict[f'{mode}/acc']
            best_acc_visual = results_dict[f'{mode}/acc_visual']
            best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']

            seed = self.cfg['train']['random_seed']
            json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')

            # save results
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)

            # print best result
            print("\nBest-----:")
            print(f'Best {mode} Acc: {best_acc:0.5f} | Visual {best_acc_visual:0.5f} | Nonvis: {best_acc_nonvis:0.5f} | Val Loss: {best_loss:0.8f} ')
            print("------------")

        # Add results to log dictionary. 
        pass#wandb.log(res, self.step_num)

        return dict(
            val_loss=res[f'{mode}/loss'],
            val_acc=res[f'{mode}/acc'],
            val_visual_acc=res[f'{mode}/acc_visual'],
            val_nonvis_acc=res[f'{mode}/acc_nonvis'],

        )

    def test_epoch_end(self, all_outputs, mode='test'):
        res = {
            'val_loss': 0.0,

            'val_correct': 0,
            'val_total': 0,

            'val_visual_correct': 0,
            'val_visual_total': 0,

            'val_nonvis_correct': 0,
            'val_nonvis_total': 0,

            'val_iou': 0.0
        }

        for output in all_outputs:
            metrics = output['metrics']
            res['val_loss'] += metrics['val_loss'].item()
            res['val_correct'] += metrics['val_correct']
            res['val_total'] += metrics['val_total']
            
            if res['val_total'] > 128:
                sanity_check = False

            res['val_visual_correct'] += metrics['val_visual_correct']
            res['val_visual_total'] += metrics['val_visual_total']

            res['val_nonvis_correct'] += metrics['val_nonvis_correct']
            res['val_nonvis_total'] += metrics['val_nonvis_total']

            if 'iou' in metrics: 
                res['val_iou'] += metrics['iou']

        res['val_loss'] = float(res['val_loss']) / len(all_outputs)
        res['val_acc'] = float(res['val_correct']) / res['val_total']
        res['val_visual_acc'] = float(res['val_visual_correct']) / res['val_visual_total']
        res['val_nonvis_acc'] = float(res['val_nonvis_correct']) / res['val_nonvis_total']

        # IoU computation. 
        res['val_iou'] = float(res['val_iou']) / (res['val_total'] * 2)

        res = {
            f'{mode}/loss': res['val_loss'],
            f'{mode}/acc': res['val_acc'],
            f'{mode}/acc_visual': res['val_visual_acc'],
            f'{mode}/acc_nonvis': res['val_nonvis_acc'],
            f'{mode}/iou': res['val_iou']
        }

        return res
