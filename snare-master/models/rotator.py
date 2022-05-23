import numpy as np
import collections
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pdb

from models.single_cls import SingleClassifier


class Rotator(SingleClassifier):

    def __init__(self, cfg):
        self.estimate_init_state = False
        self.estimate_final_state = False
        self.img_fc = None
        self.lang_fc = None
        self.cls_fc = None
        self.state_fc = None
        self.action_fc = None
        self.num_views = cfg['data']['n_views']
        super().__init__(cfg)

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

        # load pre-trained classifier (gets overrided if loading pre-trained rotator)
        # Note: gets overrided if loading pre-trained rotator
        model_path = self.cfg['train']['rotator']['pretrained_cls']
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded: {model_path}")

        self.estimate_init_state = self.cfg['train']['rotator']['estimate_init_state']
        self.estimate_final_state = self.cfg['train']['rotator']['estimate_final_state']

        # state estimation layers
        self.state_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_views)
        )

        # action layers
        self.action_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.lang_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8)
        )

        # load pre-trained rotator
        if self.cfg['train']['pretrained_model']:
            model_path = self.cfg['train']['pretrained_model']
            self.load_state_dict(torch.load(model_path)['state_dict'])
            print(f"Loaded: {model_path}")

    def forward(self, batch, teacher_force=True, init_view_force=None):
        
        # Unpack features.  
        (img1_n_feats, img2_n_feats) = batch['img_feats'] if 'img_feats' in batch else None        
        lang_feats = batch['lang_feats']
        ans = batch['ans']
        (key1, key2) =  batch['keys']
        annotation = batch['annotation']
        is_visual = batch['is_visual']

        # estimate current view
        init_state_estimation = self.estimate_state(img1_n_feats, img2_n_feats, lang_feats, init_view_force,
                                                    self.estimate_init_state)

        # output variables from state estimation
        bs = img1_n_feats.shape[0]

        img1_n_feats = init_state_estimation['img1_n_feats']
        img2_n_feats = init_state_estimation['img2_n_feats']
        lang_feats = init_state_estimation['lang_feats']

        init_views1 = init_state_estimation['init_views1']
        init_views2 = init_state_estimation['init_views2']

        est_init_views1 = init_state_estimation['est_init_views1']
        est_init_views2 = init_state_estimation['est_init_views2']

        loss = init_state_estimation['loss']

        # choose features of ramdomly sampling viewpoints
        img1_chosen_feats, img2_chosen_feats, rotated_views1, rotated_views2 = self.choose_feats_from_random_views(
            bs, img1_n_feats, img2_n_feats, init_views1, init_views2)

        # estimate second view before performing prediction
        final_state_estimation = self.estimate_state(img1_n_feats, img2_n_feats, lang_feats,
                                                     [rotated_views1, rotated_views2], self.estimate_final_state)
        est_final_views1 = final_state_estimation['est_init_views1']
        est_final_views2 = final_state_estimation['est_init_views2']
        loss += final_state_estimation['loss']

        # classifier probablities chosen features
        img1_chosen_prob = self.cls_fc(torch.cat([img1_chosen_feats, lang_feats], dim=-1))
        img2_chosen_prob = self.cls_fc(torch.cat([img2_chosen_feats, lang_feats], dim=-1))

        # classifier loss
        raw_probs = torch.cat([img1_chosen_prob, img2_chosen_prob], dim=-1)
        probs = F.softmax(raw_probs, dim=-1)
        bs = lang_feats.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_feats.device) * 2

        test_mode = (ans[0] == -1)
        if not test_mode:
            # classifier loss
            cls_labels = F.one_hot(ans)
            cls_loss_weight = self.cfg['train']['loss']['cls_weight']
            loss += (self.smoothed_cross_entropy(raw_probs, cls_labels)) * cls_loss_weight

            # put rotated views on device
            rotated_views1 = rotated_views1.to(device=self.device).int()
            rotated_views2 = rotated_views2.to(device=self.device).int()

            # state estimation accuracy
            est_init_view1_corrects = int(torch.count_nonzero(est_init_views1 == init_views1))
            est_init_view2_corrects = int(torch.count_nonzero(est_init_views2 == init_views2))
            total_correct_init_view_est = est_init_view1_corrects + est_init_view2_corrects

            est_final_view1_corrects = int(torch.count_nonzero(est_final_views1 == rotated_views1))
            est_final_view2_corrects = int(torch.count_nonzero(est_final_views2 == rotated_views2))
            total_correct_final_view_est = est_final_view1_corrects + est_final_view2_corrects

            # state estimation errors
            est_err = torch.cat([self.modulo_views(init_views1 - est_init_views1).abs().float(),
                                 self.modulo_views(init_views2 - est_init_views2).abs().float()])
            est_err += torch.cat([self.modulo_views(rotated_views1 - est_final_views1).abs().float(),
                                  self.modulo_views(rotated_views2 - est_final_views2).abs().float()])
            est_err = est_err.mean()

            return {
                'probs': probs,
                'action_loss': loss,
                'labels': cls_labels,
                'is_visual': is_visual,
                'num_steps': num_steps,

                'total_correct_init_view_est': total_correct_init_view_est,
                'total_correct_final_view_est': total_correct_final_view_est,
                'est_error': est_err,
                'est_init_views1': est_init_views1,
                'est_init_views2': est_init_views2,
                'est_final_views1': est_final_views1,
                'est_final_views2': est_final_views2,
            }
        else:
            return {
                'probs': probs,
                'num_steps': num_steps,
            }

    def estimate_state(self, img1_n_feats, img2_n_feats, lang_feats, init_view_force, perform_estimate):
        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        all_probs = []
        bs = img1_n_feats.shape[0]

        # lang encoding
        lang_feats = self.lang_fc(lang_feats)

        # normalize
        if self.cfg['train']['normalize_feats']:
            img1_n_feats /= img1_n_feats.norm(dim=-1, keepdim=True)
            img2_n_feats /= img2_n_feats.norm(dim=-1, keepdim=True)
            lang_feats /= lang_feats.norm(dim=-1, keepdim=True)

        # compute single_cls probs for 8 view pairs
        for v in range(self.num_views):
            # aggregate
            img1_feats = img1_n_feats[:, v]
            img2_feats = img2_n_feats[:, v]

            # img1 prob
            img1_feats = self.img_fc(img1_feats)
            img1_prob = self.cls_fc(torch.cat([img1_feats, lang_feats], dim=-1))

            # img2 prob
            img2_feats = self.img_fc(img2_feats)
            img2_prob = self.cls_fc(torch.cat([img2_feats, lang_feats], dim=-1))

            # cat probs
            view_probs = torch.cat([img1_prob, img2_prob], dim=-1)
            all_probs.append(view_probs)

        all_probs = torch.stack(all_probs, dim=1)
        all_probs = F.softmax(all_probs, dim=2)

        # best views with highest classifier probs
        best_views1 = all_probs[:, :, 0].argmax(-1)
        best_views2 = all_probs[:, :, 1].argmax(-1)

        # worst views with lowest classifier probs
        worst_views1 = all_probs[:, :, 0].argmin(-1)
        worst_views2 = all_probs[:, :, 0].argmin(-1)

        # Initialize with worst views
        if init_view_force == 'adv':
            init_views1 = worst_views1
            init_views2 = worst_views2
        else:
            # initialize with random views
            if init_view_force is None:
                init_views1 = torch.randint(self.num_views, (bs,)).cuda()
                init_views2 = torch.randint(self.num_views, (bs,)).cuda()
            else:
                init_views1 = init_view_force[0].to(device=self.device).int()
                init_views2 = init_view_force[1].to(device=self.device).int()

        # init features
        img1_init_feats = torch.stack([img1_n_feats[i, init_views1[i], :] for i in range(bs)])
        img2_init_feats = torch.stack([img2_n_feats[i, init_views2[i], :] for i in range(bs)])

        gt_init_views1 = F.one_hot(init_views1.to(torch.int64), num_classes=self.num_views)
        gt_init_views2 = F.one_hot(init_views2.to(torch.int64), num_classes=self.num_views)

        if perform_estimate:
            # state estimator
            est_init_views_logits1 = self.state_fc(img1_init_feats)
            est_init_views_logits2 = self.state_fc(img2_init_feats)

            # state estimation loss
            est_loss_weight = self.cfg['train']['loss']['est_weight']
            loss = ((self.smoothed_cross_entropy(est_init_views_logits1, gt_init_views1) +
                     self.smoothed_cross_entropy(est_init_views_logits2, gt_init_views2)) / 2) * est_loss_weight

            est_init_views1 = F.softmax(est_init_views_logits1, dim=-1).argmax(-1)
            est_init_views2 = F.softmax(est_init_views_logits2, dim=-1).argmax(-1)
        else:
            loss = 0
            est_init_views1 = init_views1
            est_init_views2 = init_views2

        return {
            'best_views1': best_views1,
            'best_views2': best_views2,
            'img1_n_feats': img1_n_feats,
            'img2_n_feats': img2_n_feats,
            'lang_feats': lang_feats,
            'loss': loss,
            'init_views1': init_views1,
            'init_views2': init_views2,
            'est_init_views1': est_init_views1,
            'est_init_views2': est_init_views2,
        }

    def modulo_views(self, views):
        bs = views.shape[0]
        modulo_views = torch.zeros_like(views)
        for b in range(bs):
            view = views[b]

            if view < 4 and view >= -4:
                modulo_views[b] = view
            elif view >= 4:
                modulo_views[b] = -4 + (view % 4)
            elif view < -4:
                modulo_views[b] = 4 - (abs(view) % 4)
        return modulo_views

    def choose_feats_from_random_views(self, bs, img1_n_feats, img2_n_feats, init_views1, init_views2):
        import pdb
        pdb.set_trace()
        rand_next_views = torch.randint(self.num_views, (2, bs))
        img1_chosen_feats = torch.stack([img1_n_feats[i, [init_views1[i], rand_next_views[0, i]], :].max(dim=-2)[0]
                                       for i in range(bs)])
        img2_chosen_feats = torch.stack([img2_n_feats[i, [init_views2[i], rand_next_views[1, i]], :].max(dim=-2)[0]
                                       for i in range(bs)])
        return img1_chosen_feats, img2_chosen_feats, rand_next_views[0], rand_next_views[1]

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

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, teacher_force=self.cfg['train']['rotator']['teacher_force'])

        if self.log_data:
            wandb.log({
                'tr/loss': out['action_loss'],
            })

        return dict(
            loss=out['action_loss']
        )

    def validation_step(self, batch, batch_idx):
        # view selection
        if self.cfg['val']['adversarial_init_view']:
            out = self.forward(batch, teacher_force=False, init_view_force='adv')
        else:
            bs = batch['lang_feats'].shape[0]  # get batch size off lang feats (entry index 1 in batch)
            init_view_force = [torch.ones((bs,)).int().cuda() * np.random.randint(self.num_views),
                               torch.ones((bs,)).int().cuda() * np.random.randint(self.num_views)]
            out = self.forward(batch, teacher_force=False, init_view_force=init_view_force)

        # losses
        losses = self._criterion(out)

        probs = out['probs']
        labels = out['labels']
        visual = out['is_visual']
        num_steps = out['num_steps']

        metrics = self.compute_metrics(labels, losses, probs, visual, num_steps, out)

        return dict(
            val_loss=metrics['val_loss'],
            val_acc=metrics['val_acc'],
            metrics=metrics
        )

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
