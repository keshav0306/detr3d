import torch
import lightning as L
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from diffusers import DDPMScheduler
import argparse
from lightning.pytorch.loggers import WandbLogger
from pyquaternion import Quaternion
from diffusers import UNet2DModel
from nuscenes.utils.data_classes import Box
from scipy.optimize import linear_sum_assignment
# from transformers import  AutoImageProcessor, \
#     EfficientNetModel, SwinModel, \
#          ViTModel, MobileNetV2Model

class Scheduler():
    def __init__(self, ):
        self.T = 150
        # self.betas = torch.tensor([0.5]*self.T)
        self.betas = torch.linspace(1e-3, 1e-1, steps=self.T)
        self.alphas = 1 - self.betas
        self.num_train_timesteps = self.T
        self.timesteps = torch.arange(self.T).flip(0)
        self.alpha_bars = torch.cumprod(self.alphas, 0)
    
    def add_noise(self, sample, noise, timesteps):
        mean = torch.sqrt(self.alpha_bars.to(noise)[timesteps]).to(noise)[:, None, None, None] * sample
        std = torch.sqrt(1 - self.alpha_bars.to(noise)[timesteps]).to(noise)
        return noise*std[:, None, None, None] + mean
    
    def step(self, pred_noise, t, sample):
        noise_add = torch.randn_like(sample)
        if(t != 0):
            denoised = 1/torch.sqrt(self.alphas.to(pred_noise)[t]) * (sample - self.betas.to(pred_noise)[t]/torch.sqrt(1 - self.alpha_bars.to(pred_noise)[t]) * pred_noise) + noise_add * torch.sqrt(self.betas.to(pred_noise)[t]).to(sample)
        else:
            denoised = 1/torch.sqrt(self.alphas.to(pred_noise)[t]) * (sample - self.betas.to(pred_noise)[t]/torch.sqrt(1 - self.alpha_bars.to(pred_noise)[t]) * pred_noise)
        return denoised
    
class GeneralEncoder(nn.Module):
    def __init__(self, backbone = 'resnet18', pretrained = True, num_images=1, init_ch=3):
        super(GeneralEncoder, self).__init__()
        print("inside general encoder class")
        self.backbone = backbone
        # breakpoint()
        if 'resnet' in backbone:
            self.img_preprocessor = None
            self.encoder = ResNetEncoder(backbone=backbone,
                                         pretrained=pretrained,
                                         num_images = num_images,
                                         init_ch=init_ch)
            self.encoder_dims = 512
        elif backbone == 'efficientnet':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
            self.encoder = EfficientNetModel.from_pretrained("google/efficientnet-b0") 
            self.encoder_dims = 1280
        elif backbone == 'swinmodel':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder_dims = 768
        elif backbone == 'vit':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.encoder_dims = 768
        elif backbone == 'mobilenet':
            self.encoder_dims = 1280
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.encoder = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        if 'resnet' in self.backbone:
            # print("in enc forward")
            # breakpoint()
            return self.encoder(x)
        # breakpoint()
        device = x.device
        x = self.img_preprocessor(x, return_tensors = 'pt')
        pixel_values = x['pixel_values'].to(device)
        enc_output = self.encoder(pixel_values=pixel_values)
        outputs = enc_output.last_hidden_state
        
        if self.backbone == 'vit':
            # reshaped_tensor.permute(0, 2, 1)[:,:,1:].reshape(-1, 768, 7, 7)
            reshaped_tensor = outputs.permute(0, 2, 1)[:, :, 1:].reshape(-1, 768, 14, 14)
            return reshaped_tensor
        
        if self.backbone == 'swinmodel':
            # breakpoint()
            reshaped_tensor = outputs.permute(0, 2, 1).reshape(-1, 768, 7, 7)
            return reshaped_tensor
        
        return outputs
            
        
class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_images=1, init_ch=3):
        super(ResNetEncoder, self).__init__()
        
        # Load the pre-trained ResNet model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if(num_images > 1):
            self.model.conv1 = nn.Conv2d(init_ch*num_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.model.conv1.weight.device)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

    def forward(self, x):
        # Forward pass through each ResNet block
        x = x/255
        outputs = {}
        x0 = self.layer0(x)  # First downsample: output after conv1, bn1, relu, and maxpool
        x1 = self.layer1(x0)  # Second downsample: layer1
        x2 = self.layer2(x1)  # Third downsample: layer2
        x3 = self.layer3(x2)  # Fourth downsample: layer3
        x4 = self.layer4(x3)  # Final downsample: layer4

        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] = x0, x1, x2, x3, x4
        # Return intermediate feature maps
        # breakpoint()
        return outputs[4] #downstream, only 4 is being used

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.query_dim, config.query_dim)
        self.k = nn.Linear(config.query_dim, config.query_dim)
        self.v = nn.Linear(config.query_dim, config.query_dim)
        assert config.attention_emb_dim % config.mha_heads == 0, "mha_heads must be divisible by attention_emb_dim"
        self.mha = nn.MultiheadAttention(config.attention_emb_dim, config.mha_heads, batch_first=True)
        self.out_linear = nn.Linear(config.attention_emb_dim, config.query_dim)
    
    def forward(self, q, k, v, return_attn_maps=False):
        out, attn_maps = self.mha(self.q(q), self.k(k), self.v(v), need_weights=return_attn_maps)
        # print(len(out), out[0].shape, out[1].shape)
        out = self.out_linear(out)
        if(return_attn_maps):
            return out, attn_maps
        return out

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos = nn.Parameter(torch.randn(config.max_obj, config.query_dim))
        self.norm1 = nn.LayerNorm(config.query_dim)
        self.norm2 = nn.LayerNorm(config.query_dim)
        # ca and sa block
        self.sa = Attention(config)
        self.ca = Attention(config)
        self.ff1 = nn.Linear(config.query_dim, 2*config.query_dim)
        self.ff2 = nn.Linear(2*config.query_dim, config.query_dim)
        
    def forward(self, queries, img_feats, return_attn_maps=False):
        
        queries = queries + self.pos[None]
        
        queries = self.norm1(queries)
        queries_new = self.sa(queries, queries, queries)
        queries = queries_new + queries
        
        queries = self.norm2(queries)
        if(return_attn_maps):
            queries_new, attn_maps = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        else:
            queries_new = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        queries = queries_new + queries
        queries = self.ff2(F.relu(self.ff1(queries))) + queries
        if(return_attn_maps):
            return queries, attn_maps
        return queries
    
def fourier_embedding(x, D):
    # freqs = torch.tensor([2**i for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    freqs = torch.tensor([i+1 for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    emb_sin = torch.sin(freqs * x)
    emb_cos = torch.cos(freqs * x)
    embedding = torch.cat([emb_sin, emb_cos], dim=-1)
    
    return embedding

class Det3dModel(nn.Module):
    def __init__(self, config):
        super(Det3dModel, self).__init__()
        self.config = config
        print(f"Using backbone :{config.backbone}")
        self.backbone = GeneralEncoder(config.backbone)
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.queries = nn.Parameter(torch.randn(config.max_obj, config.query_dim))
        self.pred_query_change = nn.Linear(512, config.query_dim)
        self.tr = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_tr_layers)])
        self.pred_xyz = nn.Linear(config.query_dim, 3)
        self.pred_wlh = nn.Linear(config.query_dim, 3)
        self.pred_yaw = nn.Linear(config.query_dim, 1)
        self.pred_cls = nn.Linear(config.query_dim, 2)

    def project_points(self, pts, K, img=None):
        proj = K @ pts.permute(0, 2, 1)
        proj = proj.permute(0, 2, 1)
        proj = proj / proj[..., 2:3]
        # img = np.ascontiguousarray(img.cpu().numpy().transpose(1, 2, 0))
        # for points in proj[0]:
        #     pts = points.detach().cpu().numpy().astype(np.int32)[:2]
        #     print(pts)
        #     cv2.circle(img, pts, 2, (255, 0, 255), -1)
        
        # cv2.imwrite("debug.png", img)
        # exit(0)
        proj[:, :, 0] = proj[:, :, 0] / self.config.orig_w
        proj[:, :, 1] = proj[:, :, 1] / self.config.orig_h
        
        return proj[..., :2] * 2 - 1

    def forward(self, img, K):
        b, _, _, _ = img.shape
        out = self.backbone(img)
        queries = self.queries[None].repeat(b, 1, 1)
        outputs = {}
        for i, layer in enumerate(self.tr):
            pred_xyz = self.pred_xyz(queries) # (B, N, 3)
            proj_points = self.project_points(pred_xyz, K)
            accum_feat = F.grid_sample(out, proj_points[:, None])[:, :, 0].permute(0, 2, 1) # (B, N, C)
            accum_feat = self.pred_query_change(accum_feat)
            queries = queries + accum_feat
            queries = layer(queries, queries)
        
            outputs[f'xyz_{i}'] = self.pred_xyz(queries)
            outputs[f'wlh_{i}'] = self.pred_wlh(queries)
            outputs[f'yaw_{i}'] = self.pred_yaw(queries)
            outputs[f'cls_{i}'] = self.pred_cls(queries)
        
        outputs[f'xyz'] = outputs[f'xyz_{i}']
        outputs[f'wlh'] = outputs[f'wlh_{i}']
        outputs[f'yaw'] = outputs[f'yaw_{i}']
        outputs[f'cls'] = outputs[f'cls_{i}']

        return outputs
    
    def focal_loss(self, logits, gt, alpha=2):
        probs = F.softmax(logits, -1)[torch.arange(len(logits)).to(gt), gt] + 1e-5
        loss = -(((1-probs)**alpha) * (torch.log(probs))).mean()
        return loss
    
    def make_cost_matrix(self, pred, gt):
        b, n, _ = pred['xyz'].shape
        b, m, _ = gt['xyz'].shape
        device = pred['xyz'].device
        cost_m = torch.sum((pred['xyz'][:, :, None] - gt['xyz'][:, None]) ** 2, -1)
        cost_m = cost_m.detach().cpu().numpy()
        loss = 0
        loss_dict = {'xyz': 0.0, 'wlh': 0.0, 'yaw': 0.0, 'cls': 0.0}
        total_elem = 0
        for i in range(b):
            gt_range = gt['mask'][i]
            if(gt_range == 0):
                continue
            row_ind, col_ind = linear_sum_assignment(cost_m[i, :, :gt_range])
            row_ind = torch.from_numpy(row_ind).to(device)
            col_ind = torch.from_numpy(col_ind).to(device)
            t1 = torch.arange(self.config.max_obj).to(device)
            combined = torch.cat((t1, row_ind))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            # print(pred['xyz'][i, row_ind])
            if(len(col_ind)):
                loss_dict['xyz'] += F.mse_loss(pred['xyz'][i, row_ind], gt['xyz'][i, col_ind]) * len(col_ind)/self.config.max_obj
                loss_dict['wlh'] += F.mse_loss(pred['wlh'][i, row_ind], gt['wlh'][i, col_ind]) * len(col_ind)/self.config.max_obj
                loss_dict['yaw'] += F.mse_loss(pred['yaw'][i, row_ind], gt['yaw'][i, col_ind, None]) * len(col_ind)/self.config.max_obj
                loss_dict['cls'] += self.focal_loss(pred['cls'][i, row_ind], torch.ones(len(row_ind)).long().to(device)) * len(col_ind)/self.config.max_obj

                # print("XYZ : ", pred['xyz'][i, row_ind], gt['xyz'][i, col_ind])
                # print("WLH : ", pred['wlh'][i, row_ind], gt['wlh'][i, col_ind])
                # print("YAW : ", pred['yaw'][i, row_ind], gt['yaw'][i, col_ind])
                # print("CLS : ", pred['cls'][i, row_ind])
                
            if(len(difference)):
                # loss_elem += 0.1 * (F.cross_entropy(pred['cls'][i, difference], torch.zeros(len(difference)).long().to(device))) * len(difference)
                # loss_elem += (self.focal_loss(pred['cls'][i, difference], torch.zeros(len(difference)).long().to(device))) * len(difference)
                loss_dict['cls'] += (self.focal_loss(pred['cls'][i, difference], torch.zeros(len(difference)).long().to(device))) * len(difference)/self.config.max_obj
                # print("CLS REMAIN: ", pred['cls'][i, difference])

            assert torch.isnan(loss_dict['cls']).sum() == 0, f"{col_ind} {row_ind} {gt_range} {len(pred)} {len(gt)} {difference}"
            # loss += self.compute_aux_loss(pred, gt, row_ind, col_ind, i, difference, device)

        loss = torch.zeros(1, requires_grad=True).to(device)
        for k, v in loss_dict.items():
            loss += v
            
        return loss/b, loss_dict

    def compute_aux_loss(self, pred, gt, row_ind, col_ind, idx, difference, device):
        loss = 0.0
        for i in range(self.config.num_tr_layers - 1):
            if(len(col_ind)):
                loss_elem = (F.mse_loss(pred[f'xyz_{i}'][idx, row_ind], gt['xyz'][idx, col_ind]) + \
                            F.mse_loss(pred[f'wlh_{i}'][idx, row_ind], gt['wlh'][idx, col_ind]) + \
                            F.mse_loss(pred[f'yaw_{i}'][idx, row_ind], gt['yaw'][idx, col_ind, None]) + \
                            # 0.1 * F.cross_entropy(pred['cls'][i, row_ind], torch.ones(len(row_ind)).long().to(device))) * len(col_ind)
                            self.focal_loss(pred[f'cls_{i}'][idx, row_ind], torch.ones(len(row_ind)).long().to(device))) * len(col_ind)
            else:
                loss_elem = 0.0
            if(len(difference)):
                # loss_elem += 0.1 * (F.cross_entropy(pred['cls'][i, difference], torch.zeros(len(difference)).long().to(device))) * len(difference)
                loss_elem += (self.focal_loss(pred[f'cls_{i}'][idx, difference], torch.zeros(len(difference)).long().to(device))) * len(difference)

            loss += loss_elem
        
        return loss

    def compute_loss(self, batch, is_train=True):
        output = {}
        data = batch
        img = data['img']
        K = data['K']
        # which = 1
        # self.project_points(data['xyz'][which, :data['mask'][which]][None], data['K'][which][None], data['img'][which])
        pred_outputs = self(img, K)
        loss, loss_dict = self.make_cost_matrix(pred_outputs, data)
        output['loss'] = loss
        output['loss_dict'] = loss_dict
        return output
    
    def validate(self, batch):
        output = self.compute_loss(batch, is_train=False)
        return output
    
    def infer(self, img, K, gt=None):
        b, _, h, w = img.shape
        output = self(img, K)
        cls_probs = F.softmax(output['cls'], -1)
        print(cls_probs)
        cls_mask = torch.argmax(cls_probs, -1) == 1 
        # cls_mask = torch.argmax(cls_probs, -1) != 3
        all_boxes = []
        for i in range(b):
            pred_xyz = output['xyz'][i][cls_mask[i]].cpu().numpy()
            pred_wlh = output['wlh'][i][cls_mask[i]].cpu().numpy()
            pred_yaw = output['yaw'][i][cls_mask[i]].cpu().numpy()
            # gt_xyz = gt['xyz'][i, :gt['mask'][i]].cpu().numpy()
            # gt_wlh = gt['wlh'][i, :gt['mask'][i]].cpu().numpy()
            # gt_yaw = gt['yaw'][i, :gt['mask'][i]].cpu().numpy()
            # print(gt_xyz)
            # print(pred_xyz)
            # print(gt_wlh)
            # print(pred_wlh)
            # print(gt_yaw)
            # print(pred_yaw)
            # print()
            boxes = []
            for j in range(len(pred_yaw)):
                pred_quat = Quaternion(axis=[0, 1, 0], angle=pred_yaw[j])
                boxes.append(Box(pred_xyz[j], pred_wlh[j], pred_quat))
            all_boxes.append(boxes)
        # exit(0)
        return all_boxes


class LITDetModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Det3dModel(config)
        
    def training_step(self, batch, idx):
        # compute the training loss and log it to wandb and return it as well to update the model
        output = self.model.compute_loss(batch)
        train_loss = output['loss']
        
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        # for k, v in output['loss_dict'].items():
        #     print(k, v)
        # print("loss : ", train_loss)
        return train_loss
    
    def validation_step(self, batch, idx):
        # log the validation_loss, visualization images to wandb
        data = batch
        output = self.model.validate(data)
        self.log("val_loss", output['loss'], sync_dist=True, prog_bar=True)
        if(idx == 0):
            pred_boxes = self.model.infer(data['img'], data['K'], data)
            vis_imgs = self.visualize(data['img'], pred_boxes, data['K'])
            for i, img in enumerate(vis_imgs):
                cv2.imwrite(f"vis/{i}.png", img)
    
    def visualize(self, img, pred_boxes, K):
        imgs = img.permute(0, 2, 3, 1).cpu().numpy()
        K = K.cpu().numpy()
        all_vis = []
        for (img, box, K_) in zip(imgs, pred_boxes, K):
            img = np.ascontiguousarray(img.astype(np.uint8))
            img = cv2.resize(img, (1600, 900))
            for b in box:
                b.render_cv2(img, view=K_, normalize=True)
            all_vis.append(img)
            
        return all_vis

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)
        return optimizer