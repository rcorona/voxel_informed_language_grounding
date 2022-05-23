from omegaconf import DictConfig
import torch
import torch.nn as nn
import pdb
import os

from legoformer.model.base import BaseModel
from legoformer.model.transformer import Transformer
from legoformer.model.output import OutputLayer
import legoformer.util.binvox_rw as binvox_rw

class LegoFormer(BaseModel):
    """
        Base class for the LegoFormer models.
        Contains shared logic between multi-view and single-view models.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        # Grab config specific to the overall network & transformer
        self.cfg_optim = config.optimization
        self.cfg_network = config.network
        self.cfg_transformer = self.cfg_network.transformer
        self.cfg_eval = config.eval
        self.img_source = config.data.dataset.ShapeNet.img_source
        self.cfg_data = config.data

        # Grab config values
        self.d_model            = self.cfg_transformer.d_model
        self.n_vox              = config.data.constants.n_vox
        self.num_queries        = self.cfg_network.n_queries
        self.clip_output        = self.cfg_network.clip_output

        # Initialize learned queries, transformer & output layer
        self.learned_queries    = nn.Parameter(torch.rand((self.num_queries, self.d_model)))#.cuda()
        self.transformer        = Transformer(**self.cfg_transformer)
        self.output_layer       = OutputLayer(self.d_model, self.n_vox)

    def training_step(self, batch, batch_idx):
        """
            Called from PyTorch-Lightning framework
                and performs a single training step (forward pass + loss calculation)
        :param batch: Collocated batch data returned by DataLoader.
                        Contains images, GT volumes and info about samples in the batch
        :param batch_idx: Batch ID (integer)
        :return: The loss calculated in the `run_step` method
        """
        return self.run_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """
            Called from PyTorch-Lightning framework
                and performs a single validation step
        :param batch: Collocated batch data returned by DataLoader.
                        Contains images, GT volumes and info about samples in the batch
        :param batch_idx: Batch ID (integer)
        :return: The loss calculated in the `run_step` method
        """
        return self.run_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """
            Called from PyTorch-Lightning framework
                and performs a single test step
        :param batch: Collocated batch data returned by DataLoader.
                        Contains images, GT volumes and info about samples in the batch
        :param batch_idx: Batch ID (integer)
        :return: The loss calculated in the `run_step` method
        """
        return self.run_step(batch, 'test')

    def run_step(self, batch, tag):
        """
            Shared logic for the train | validate | test phase forward step
        :param batch: Collocated batch data returned by DataLoader.
                        Contains images, GT volumes and info about samples in the batch
        :param tag: Phase tag. Should be one of the <train | val | test>
        :return: Calculated loss
        """
        # extract batch
        # image.shape => [B, N, 3, H (224), W (224)]
        # gt_volume.shape => [B, 32, 32, 32]
        image, gt_volume, info = batch

        # images => decomposition factors
        output = self.forward(image)  # output is a dict

        # decomposition factors => 3D voxel grid
        # pred_volume.shape => [B, 32, 32, 32]
        pred_volume = self.aggregate(output)

        loss = self.calculate_loss(pred_volume, gt_volume)

        if tag == 'test' and self.cfg_eval.save_voxels: 
            self.save_voxels(pred_volume, gt_volume, info, self.cfg_eval.save_dir, self.img_source)

        # Log loss & metrics (IoU + F1_score). F1_score will be logged only if tag=='test'
        self.log_scalar(f'loss/{tag}', loss)
        self.log_metrics(pred_volume, gt_volume, tag)
        return loss

    def save_voxels(self, pred_volume, gt_volume, info, save_dir, img_source): 
        
        # Binarize so we can read volume. 
        gt_volume = self.binarize_preds(gt_volume).detach().cpu().numpy()
        pred_volume = self.binarize_preds(pred_volume, 0.3).detach().cpu().numpy()

        for i in range(len(info[0])):

            # Unpack data. 
            taxonomy = info[0][i]
            obj = info[1][i]
            pred = pred_volume[i]
            gt = gt_volume[i]

            # Paths to binvox files. 
            gt_path = os.path.join(save_dir, '{}_gt.binvox'.format(obj))
            pred_path = os.path.join(save_dir, '{}_pred_{}.binvox'.format(obj, img_source))
            sanity_path = os.path.join(save_dir, '{}_sanity.binvox'.format(obj)) 

            # Load original gt binvox file to get necessary meta data. 
            original_path = self.cfg_data['dataset']['ShapeNet']['voxel_path'] % (taxonomy, obj)

            with open(original_path, 'rb') as f:
                volume = binvox_rw.read_as_3d_array(f) 
            
            # Save files. 
            volume.write(open(sanity_path, 'wb'))

            volume.data = gt
            volume.write(open(gt_path, 'wb'))

            volume.data = pred
            volume.write(open(pred_path, 'wb'))

    def get_obj_features(self, vgg16_feats, use_xyz=False, reconstruction=False): 
        """
        Get only object query output from legoformer without projecting. 
        return: object queries embedded by transformer. 
        """

        # ------------ PRE-TRANSFORMER PART ------------
        b_size = vgg16_feats.size(0)

        # Project VGG16 features into token space if in multiview. 
        if self.view_mode == 'multi': 
            inp_tokens = self.view_embedder(vgg16_feats)
        else: 
            inp_tokens = vgg16_feats.squeeze()

        # prepare inputs for the first decoder layer
        tgt = self.prepare_learned_queries(self.learned_queries, b_size)
        # prepare decoder-side attention mask
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        # [B, T, D] => [T, B, D]  : B-batch size, T-num decomposition factors, D-model dimensionality
        inp_tokens = inp_tokens.permute((1, 0, 2))  # shape: h*w, b_size, n_views * n_ch
        
        # run transformer
        prediction = self.transformer(inp_tokens, tgt, decoder_attn_mask)
        # [T, B, D] => [B, T, D]
        prediction = prediction.permute((1, 0, 2))

        # TODO might be better to output final layer? 
        # Map transformer output to decomposition factors
        z_factors, y_factors, x_factors = self.output_layer(prediction)

        if use_xyz: 
            output = torch.cat([z_factors, y_factors, x_factors], dim=-1)
        else: 
            output = prediction

        # Additionally get reconstruction if desired. 
        if reconstruction: 

            # Full reconstruction. 
            part_volumes = torch.einsum('bni,bnj,bnk->bnijk', (z_factors, y_factors, x_factors))
            predicted_volume = torch.einsum('bni,bnj,bnk->bijk', (z_factors, y_factors, x_factors))

            if self.clip_output: 
                part_volumes = part_volumes.clamp_max(1.0)
                predicted_volume = predicted_volume.clamp_max(1.0)
    
                reconstruction_output = {'part_volumes': part_volumes, 'full_volume': predicted_volume}

        else: 
            reconstruction_output = None
 
        return output, reconstruction_output

    def forward(self, images):
        """
            Forward pass logic
        :param images: Input images
        :return: Decomposition factors
        """
        images = images.to('cuda')
        
        # ------------ PRE-TRANSFORMER PART ------------

        # retrieve batch size
        b_size = images.shape[0]
        # embed input images in input tokens (vectors)
        inp_tokens = self.images2tokens(images)
        # prepare inputs for the first decoder layer
        tgt = self.prepare_learned_queries(self.learned_queries, b_size)
        # prepare decoder-side attention mask
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        # [B, T, D] => [T, B, D]  : B-batch size, T-num decomposition factors, D-model dimensionality
        inp_tokens = inp_tokens.permute((1, 0, 2))  # shape: h*w, b_size, n_views * n_ch
        # run transformer
        prediction = self.transformer(inp_tokens, tgt, decoder_attn_mask)
        # [T, B, D] => [B, T, D]
        prediction = prediction.permute((1, 0, 2))

        # ------------ POST-TRANSFORMER (OUTPUT) PART ------------

        # Map transformer output to decomposition factors
        z_factors, y_factors, x_factors = self.output_layer(prediction)

        return {
            'decomposition_factors': {
                'z': z_factors,
                'y': y_factors,
                'x': x_factors
            }
        }

    def images2tokens(self, images):
        """
            Maps input images into input tokens for the Transformer-Encoder.
        :param images: Input Images
        :return:
        """
        raise NotImplementedError('Should be implemented in the subclass!')

    def get_decoder_mask(self):
        """
            Generate decoder-side attention mask
        :return: Attention mask or None
        """
        raise NotImplementedError('Should be implemented in the subclass!')

    def aggregate(self, output: dict):
        """
            Compute 3D voxel grid out of the decomposition factors
        :param output:
        :return:
        """
        # Extract decomposition factors
        z_factors = output['decomposition_factors']['z']
        y_factors = output['decomposition_factors']['y']
        x_factors = output['decomposition_factors']['x']

        # Given a set of triple vectors [B x 3 x n_vox], the einsum expression here works as follows,
        # 1) Take a cross-product between each triple vectors to obtain a set of 3D tensor
        # 2) Sum 3D tensors to obtain a single 3D tensor
        predicted_volume = torch.einsum('bni,bnj,bnk->bijk', (z_factors, y_factors, x_factors))

        if self.clip_output:
            # Clip output volume
            predicted_volume = predicted_volume.clamp_max(1)
        return predicted_volume
