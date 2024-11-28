import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import bev_utils
import bev_utils.geom
import bev_utils.basic
import bev_utils.vox
import bev_utils.misc

from .encoder import Encoder_res101, Encoder_dinov2
from .decoder import Decoder


EPS = 1e-4

class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None, 
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_args=None,
                 ):
        super(Segnet, self).__init__()


        self.Z, self.Y, self.X = Z, Y, X
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_args['encoder_type']

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if self.encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif "dino" in self.encoder_type:
            self.encoder = Encoder_dinov2(feat2d_dim, 
                                          encoder_args
                                          )

        # BEV compressor
        if self.do_rgbcompress:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            # use simple sum
            pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if vox_util is not None:
            self.xyz_memA = bev_utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: bev_utils.basic.pack_seqdim(x, B)
        __u = lambda x: bev_utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = bev_utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device  # (B*S, C, H, W)
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])

        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = bev_utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            bev_utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = bev_utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

        # bev compressing
        if self.do_rgbcompress:
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e

