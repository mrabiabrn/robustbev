
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import bev_utils as utils
from .segnet import Segnet


class SimpleBEV(nn.Module):
    def __init__(self, args):
        super(SimpleBEV, self).__init__()
        self.args = args
        
        do_rgbcompress = args.do_rgbcompress
        rand_flip = args.rand_flip
        
        # default values of SimpleBEV
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid_py).float()

        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        YMIN, YMAX = -5, 5
        self.bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

        self.Z, self.Y, self.X = 200, 8, 200 

        vox_util = utils.vox.Vox_util(
                                            self.Z, self.Y, self.X,
                                            scene_centroid=self.scene_centroid.cuda(),
                                            bounds=self.bounds,
                                            assert_cube=False)
        
        self.model = Segnet(
                            self.Z, self.Y, self.X, vox_util, 
                            do_rgbcompress=do_rgbcompress, 
                            rand_flip=rand_flip,
                            encoder_args=args.encoder_args,
                            )
        
        self.loss_fn = SimpleLoss(2.13).cuda()
        
    def forward(self, batch):

        imgs, rots, trans, intrins, seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g = batch

        B,S,C,H,W = imgs.shape
        device = imgs.device

        intrins_ = intrins.reshape(B*S, 4, 4)
        pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
        pix_T_cams = pix_T_cams_.reshape(B, S, 4, 4)

        velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
        cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
        
        vox_util = utils.vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid.to(device),
            bounds=self.bounds,
            assert_cube=False)
        
        _, _, seg_bev_e, center_bev_e, offset_bev_e = self.model(
                                                                    rgb_camXs=imgs,
                                                                    pix_T_cams=pix_T_cams,
                                                                    cam0_T_camXs=cam0_T_camXs,
                                                                    vox_util=vox_util
                                                                    )
        

        seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum(dim=[1,2,3])
        union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum(dim=[1,2,3])
        iou = (intersection/(1e-4 + union)).mean()
        

        total_loss = torch.tensor(0.0).cuda()

        ce_loss = self.loss_fn(seg_bev_e, seg_bev_g, valid_bev_g)

        center_loss = balanced_mse_loss(center_bev_e, center_bev_g)
        offset_loss = torch.abs(offset_bev_e-offset_bev_g).sum(dim=1, keepdim=True)
        offset_loss = utils.basic.reduce_masked_mean(offset_loss, seg_bev_g*valid_bev_g)

        ce_factor = 1 / torch.exp(self.model.ce_weight)
        ce_loss = 10.0 * ce_loss * ce_factor
        ce_uncertainty_loss = 0.5 * self.model.ce_weight

        center_factor = 1 / (2*torch.exp(self.model.center_weight))
        center_loss = center_factor * center_loss
        center_uncertainty_loss = 0.5 * self.model.center_weight

        offset_factor = 1 / (2*torch.exp(self.model.offset_weight))
        offset_loss = offset_factor * offset_loss
        offset_uncertainty_loss = 0.5 *self.model.offset_weight

        total_loss += ce_loss
        total_loss += center_loss
        total_loss += offset_loss
        total_loss += ce_uncertainty_loss
        total_loss += center_uncertainty_loss
        total_loss += offset_uncertainty_loss

        loss_details = {
            'ce': ce_loss.item(),
            'center': center_loss.item(),
            'offset': offset_loss.item(),
            'ce_uncertainty': ce_uncertainty_loss.item(),
            'center_uncertainty': center_uncertainty_loss.item(),
            'offset_uncertainty': offset_uncertainty_loss.item(),
        }

        return {
            'total_loss': total_loss,
            'loss_details': loss_details,
            'iou': iou,
            'intersection': intersection,
            'union': union,
        }
    



class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss
    
