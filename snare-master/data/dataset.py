import os
import json
import torch
import torch.utils.data
import clip
import cv2
import torchvision.models

import numpy as np
import gzip
import json
import pdb
import tqdm
from einops import rearrange

import legoformer.data as transforms
from legoformer.data.dataset import ShapeNetDataset
from data.verify_shapenet import get_snare_objs

class CLIPGraspingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode='train', legoformer_data_module=None):
        self.total_views = 14
        self.cfg = cfg
        self.mode = mode
        self.folds = os.path.join(self.cfg['data']['amt_data'], self.cfg['data']['folds'])
        self.feats_backbone = self.cfg['train']['feats_backbone']

        self.n_views = self.cfg['data']['n_views']

        print("Num views: {}".format(self.n_views))

        self.load_entries()
        self.load_extracted_features()

        # Paths to ShapeNet rendered images. 
        self.shapenet_path = os.path.join(self.cfg['root_dir'], 'data/screenshots') 

        # Use images during loading or not (if feeding straight into LegoFormer). 
        if self.cfg['train']['model'] == 'transformer': 
            self.use_imgs = True
        else: 
            self.use_imgs = False

        # Get transforms for preprocessing ShapeNet images. 
        if legoformer_data_module: 
            self.transforms = legoformer_data_module.get_eval_transforms(legoformer_data_module.cfg_data.transforms)

    def preprocess_obj_feats(self): 

        # Don't need to pre-extract legoformer features. 
        if self.feats_backbone == 'legoformer':
            return 

        # Chose model. 
        if self.feats_backbone == 'pix2vox': 
            model = Pix2Vox(self.cfg)
        elif self.feats_backbone == '3d-r2n2': 
            raise NotImplementedError

        # First make set of all objects that don't yet have a feature matrix. 
        missing_objs = set()
        done_objs = set()

        for obj in os.listdir(self.shapenet_path): 
            
            # Single or multi-view.
            file_path = os.path.join(self.shapenet_path, obj, '{}-{}-{}.npy'.format(obj, self.feats_backbone, self.n_views))

            if not os.path.isfile(file_path):
                missing_objs.add(obj)
            else: 
                done_objs.add(obj)

        # This is all the objects. 
        if len(done_objs) >= 7881:
            return

        # Intersect with list of objects actually in dataset. 
        # Note: we can use this same file in both single and multiview, is just a sanity check. 
        snare_objs = get_snare_objs()

        objs = list(snare_objs & missing_objs)
        obj_feats_dict = None

        print('Extracting {} {}-view features for {} objects...'.format(self.feats_backbone, self.n_views, len(objs)))

        # Used to load images quickly. 
        class ObjImgDataset(torch.utils.data.Dataset): 

            def __init__(self, objs, snare_dataset):
                self.objs = objs
                self.snare_dataset = snare_dataset

            def __getitem__(self, idx): 
                item = dict()
                item['images'] = self.snare_dataset.get_imgs(self.objs[idx])
                item['idx'] = idx

                return item

            def __len__(self): 
                return len(self.objs)

        # Now go through objects to extract backbone features. 
        obj_dataset = ObjImgDataset(objs, self)
        bz = 8
        dataloader = torch.utils.data.DataLoader(obj_dataset, batch_size=bz, num_workers=32)

        for b_idx, batch in enumerate(tqdm.tqdm(dataloader)): 
            
            # Get backbone features and prep for input to LegoFormer view embedder (which we'll finetune). 
            imgs = batch['images']

            with torch.no_grad(): 

                if self.n_views > 1:  
                    backbone_feats = model.get_intermediate_feats(imgs) 
                    backbone_feats = backbone_feats.view(imgs.size(0), -1)
                else: 
                    # Pass each image as its own data example. 
                    backbone_feats = model.get_intermediate_feats(imgs.view(imgs.size(0) * 8, 1, 3, 224, 224))
                    backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
                    backbone_feats = backbone_feats.view(imgs.size(0), 8, backbone_feats.size(-1))

                # TODO need single view case! Can probably just squash over batch dimension and treat each img as its own datapoint. 

            # Now iterate over all objects in batch to store them. 
            for i in range(backbone_feats.size(0)):
                obj_id = obj_dataset.objs[batch['idx'][i]]
                feats = backbone_feats[i].detach().cpu().numpy()
                npy_path = os.path.join(self.shapenet_path, obj_id, '{}-{}-{}.npy'.format(obj_id, self.feats_backbone, self.n_views))

                # Store as npz file. 
                np.save(npy_path, feats) 

    def preprocess_vgg16(self, legoformer_model): 

        # First make set of all objects that don't yet have a feature matrix. 
        missing_objs = set()
        done_objs = set()

        for obj in os.listdir(self.shapenet_path): 
            
            # If skipping legoformer, collect raw vgg16 features altogether without using legoformer. 
            if self.cfg['transformer']['skip_legoformer']: 
                file_path = os.path.join(self.shapenet_path, obj, '{}-rawVGG.npy'.format(obj))

            # Single or multi-view. 
            elif self.n_views == 1: 
                file_path = os.path.join(self.shapenet_path, obj, '{}-single.npy'.format(obj))
            else: 
                file_path = os.path.join(self.shapenet_path, obj, '{}.npy'.format(obj))

            if not os.path.isfile(file_path):
                missing_objs.add(obj)
            else: 
                done_objs.add(obj)

        # This is all the objects. 
        if len(done_objs) >= 7881:
            return

        # Intersect with list of objects actually in dataset. 
        # Note: we can use this same file in both single and multiview, is just a sanity check. 
        snare_objs = get_snare_objs()

        objs = list(snare_objs & missing_objs)
        obj_feats_dict = None

        print('Extracting VGG16 features for {} objects...'.format(len(objs)))

        # Used to load images quickly. 
        class ObjImgDataset(torch.utils.data.Dataset): 

            def __init__(self, objs, snare_dataset):
                self.objs = objs
                self.snare_dataset = snare_dataset

            def __getitem__(self, idx): 
                item = dict()
                item['images'] = self.snare_dataset.get_imgs(self.objs[idx])
                item['idx'] = idx

                return item

            def __len__(self): 
                return len(self.objs)

        # Now go through objects to extract backbone features. 
        obj_dataset = ObjImgDataset(objs, self)
        bz = 8
        dataloader = torch.utils.data.DataLoader(obj_dataset, batch_size=bz, num_workers=32)

        # If skipping LegoFormer, then just use pre-trained VGG16 itself (same as used by LegoFormer though). 
        if self.cfg['transformer']['skip_legoformer']:
            vgg16 = torchvision.models.vgg16(pretrained=True)
            vgg16.classifier = vgg16.classifier[:-1]         
            vgg16.cuda()

        for b_idx, batch in enumerate(tqdm.tqdm(dataloader)): 
            
            # Get backbone features and prep for input to LegoFormer view embedder (which we'll finetune). 
            imgs = batch['images']

            with torch.no_grad(): 

                if self.cfg['transformer']['skip_legoformer']:
                    backbone_feats = imgs.view(imgs.size(0) * 8, 3, 224, 224).cuda()
                    backbone_feats = vgg16(backbone_feats).view(imgs.size(0), 8, -1)

                # Single-view processing. 
                elif self.n_views == 1: 
                    imgs = imgs.view(imgs.size(0) * 8, 1, 3, 224, 224)
                    feats = legoformer_model.legoformer.backbone(imgs)
                    patches = legoformer_model.legoformer.split_features(feats)
                    patches = legoformer_model.legoformer.add_2d_pos_enc(patches)
                    backbone_feats = rearrange(patches, 'b n np d -> b (n np) d')
                    backbone_feats = backbone_feats.reshape(imgs.size(0) // 8, 8, backbone_feats.size(1), backbone_feats.size(-1))
                else:
                    backbone_feats = legoformer_model.legoformer.backbone(imgs)
                    backbone_feats = rearrange(backbone_feats, 'b n c h w -> b n (c h w)')

            # Now iterate over all objects in batch to store them. 
            for i in range(backbone_feats.size(0)):
                obj_id = obj_dataset.objs[batch['idx'][i]]
                feats = backbone_feats[i].detach().cpu().numpy()

                if self.cfg['transformer']['skip_legoformer']:
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}-rawVGG.npy'.format(obj_id))

                    assert feats.shape == (8, 4096)

                # Single or multiview case.
                elif self.n_views == 1: 
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}-single.npy'.format(obj_id))
                else: 
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}.npy'.format(obj_id))

                # Store as npz file. 
                np.save(npy_path, feats) 

    def load_entries(self):
        train_train_files = ["train.json"]
        train_val_files = ["val.json"]
        test_test_files = ["test.json"]

        # modes
        if self.mode == "train":
            self.files = train_train_files
        elif self.mode  == 'valid':
            self.files = train_val_files
        elif self.mode == "test":
            self.files =  test_test_files
        else:
            raise RuntimeError('mode not recognized, should be train, valid or test: ' + str(self.mode))

        # load amt data
        self.data = []
        for file in self.files:
            fname_rel = os.path.join(self.folds, file)
            print(fname_rel)
            with open(fname_rel, 'r') as f:
                self.data = self.data + json.load(f)

        print(f"Loaded Entries. {self.mode}: {len(self.data)} entries")

    def load_extracted_features(self):

        # Determine which features to use. 
        self.use_lang_feats = self.cfg['train']['model'] != 'transformer'
        self.use_img_feats = True# TODO = self.feats_backbone == "clip" or self.feats_backbone == 'multimodal' 
        
        # Load pre-trained CLIP language features if not using transformer model.  
        if self.use_lang_feats: 
            lang_feats_path = self.cfg['data']['clip_lang_feats']
            with open(lang_feats_path, 'r') as f:
                self.lang_feats = json.load(f)

        # Load pre-trained image embeddings if desired.  
        if self.use_img_feats: 
            with open(self.cfg['data']['clip_img_feats'], 'r') as f:     
                self.img_feats = json.load(f)

    def __len__(self):
        # Accomodate for larger dataset if different combos are possible. 
        if self.n_views != 8 and self.mode != 'train': 
            return len(self.data) * 10
        else: 
            return len(self.data)

    def get_img_feats(self, key):

        img_feats = []
        for i in range(self.total_views):
            feat = np.array(self.img_feats[f'{key}-{i}'])
            img_feats.append(feat)

        return np.array(img_feats)

    def get_obj_feats(self, obj): 
        file_path = os.path.join(self.shapenet_path, obj, '{}-{}-{}.npy'.format(obj, self.feats_backbone, self.n_views))

        return np.load(file_path)

    def get_imgs(self, key): 
        
        # Object images path. 
        img_dir = os.path.join(self.shapenet_path, key)

        # Iterate over images and load them. 
        imgs = []
        img_idxs = np.arange(self.total_views)[6:]# TODO we hardcode the standard 8-views for now. 

        for idx in img_idxs: 
            img_path =  os.path.join(img_dir, '{}-{}.png'.format(key, idx))
            img = ShapeNetDataset.read_img(img_path)
            imgs.append(img)

        imgs = np.asarray(imgs)

        # Add transformations from LegoFormer. THIS STEP IS CRUCIAL.
        imgs = self.transforms(imgs)

        return imgs

    def get_vgg16_feats(self, key): 

        # Object images path. 
        feat_dir = os.path.join(self.shapenet_path, key)

        if self.cfg['transformer']['skip_legoformer']:
            feat_path = os.path.join(feat_dir, '{}-rawVGG.npy'.format(key))

        # Single-view features. 
        elif self.n_views == 1: 
            feat_path = os.path.join(feat_dir, '{}-single.npy'.format(key))
        
        # Multi-view features. 
        else: 
            feat_path = os.path.join(feat_dir, '{}.npy'.format(key))
        
        return np.load(feat_path)

    def __getitem__(self, idx):

        if self.cfg['train']['tiny_dataset']:
            idx = idx % self.cfg['train']['batch_size']
        
        # Add more examples for stability if different view combinations are possible. 
        if self.n_views != 8 and self.mode != 'train': 
            idx = idx // 10

        # Will return features in dictionary form. 
        feats = dict()
        entry = self.data[idx]

        # get keys
        entry_idx = entry['ans'] if 'ans' in entry else -1 # test set does not contain answers
        if len(entry['objects']) == 2:
            key1, key2 = entry['objects']

        # fix missing key in pair by sampling alternate different object from data. 
        else:
            key1 = entry['objects'][entry_idx]
 
            while True:

                alt_entry = self.data[np.random.choice(len(self.data))]
                key2 = np.random.choice(alt_entry['objects'])

                if key2 != key1:
                    break
        
        # annotation
        annotation = entry['annotation']
        feats['annotation'] = annotation

        # test set does not have labels for visual and non-visual categories
        feats['is_visual'] = entry['visual'] if 'ans' in entry else -1

        # Select view indexes randomly # TODO need to select them consistently for evaluation.
        view_idxs1 = np.random.choice(8, self.n_views, replace=False)
        view_idxs2 = np.random.choice(8, self.n_views, replace=False)

        ## Img feats
        # For CLIP filter to use only desired amount of views (in this case 8). 
        if self.use_img_feats: 
            start_idx = 6 # discard first 6 views that are top and bottom viewpoints
            img1_n_feats = torch.from_numpy(self.get_img_feats(key1))[start_idx:]
            img2_n_feats = torch.from_numpy(self.get_img_feats(key2))[start_idx:]
      
            # Pick out sampled views. 
            img1_n_feats = img1_n_feats[view_idxs1]
            img2_n_feats = img2_n_feats[view_idxs2]

            feats['img_feats'] = (img1_n_feats, img2_n_feats)

        # Object reconstruction model features, except for legoformer. 
        if self.feats_backbone == 'pix2vox' or self.feats_backbone == '3d-r2n2':
            obj1_n_feats = torch.from_numpy(self.get_obj_feats(key1))
            obj2_n_feats = torch.from_numpy(self.get_obj_feats(key2))
           
            if self.n_views == 1: 
                obj1_n_feats = obj1_n_feats[view_idxs1]
                obj2_n_feats = obj2_n_feats[view_idxs2]

            feats['obj_feats'] = (obj1_n_feats, obj2_n_feats)

        # Tokenize annotation if using a transformer. 
        if self.cfg['train']['model'] == 'transformer':
            feats['lang_tokens'] = clip.tokenize(feats['annotation'])
        else: 
            feats['lang_feats'] = torch.from_numpy(np.array(self.lang_feats[annotation]))

        # label
        feats['ans'] = entry_idx
    
        # Keys
        feats['keys'] = (key1, key2)

        # Return VGG16 features if needed.
        if self.feats_backbone == 'legoformer': 
            vgg16_feats1 = self.get_vgg16_feats(key1)
            vgg16_feats2 = self.get_vgg16_feats(key2)

            # Filter out views. 
            vgg16_feats1 = vgg16_feats1[view_idxs1]
            vgg16_feats2 = vgg16_feats2[view_idxs2]

            feats['vgg16_feats'] = (vgg16_feats1, vgg16_feats2)

        # Load ground truth voxel maps if needed. 
        if self.cfg['data']['voxel_reconstruction']:
            volume1_path = os.path.join(self.cfg['data']['shapenet_voxel_dir'], '{}-32.npy'.format(key1))
            volume2_path = os.path.join(self.cfg['data']['shapenet_voxel_dir'], '{}-32.npy'.format(key2))

            # Skip voxel maps we don't have good data for. 
            if os.path.isfile(volume1_path):
                volume1 = np.load(volume1_path)
                volume1_mask = 1.0
            else: 
                volume1 = np.zeros((32, 32, 32), dtype=np.float32)
                volume1_mask = 0.0

            if os.path.isfile(volume2_path):
                volume2 = np.load(volume2_path)
                volume2_mask = 1.0
            else: 
                volume2 = np.zeros((32, 32, 32), dtype=np.float32)
                volume2_mask = 0.0

            feats['voxel_maps'] = (volume1, volume2)
            feats['voxel_masks'] = (volume1_mask, volume2_mask)

        # TODO Probably don't want this since it slows dataloading. But may need for direct input at some point? 
        # Additionally return images, either as input or for visualization purposes. 
        """
        imgs1 = self.get_imgs(key1)
        imgs2 = self.get_imgs(key2)

        feats['images'] = (imgs1, imgs2)
        """

        return feats
