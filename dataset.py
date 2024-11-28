"""
    Adapted from https://github.com/aharley/simple_bev
"""


import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box

import torchvision

import bev_utils.py
import bev_utils.geom
import bev_utils.vox


discard_invisible = False


def move_refcam(data, refcam_id):

    data_ref = data[refcam_id].clone()
    data_0 = data[0].clone()

    data[0] = data_ref
    data[refcam_id] = data_0

    return data

def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img

totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class NuscData(torch.utils.data.Dataset):

    def __init__(self, nusc, 
                 is_train, 
                 data_aug_conf, 
                 centroid=None, 
                 bounds=None, 
                 res_3d=None, 
                 refcam_id=1, 
                 do_shuffle_cams=True,
                 ):
        
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.do_shuffle_cams = do_shuffle_cams
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.refcam_id = refcam_id

        self.dataroot = self.nusc.dataroot
            
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()


        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
        Z, Y, X = self.res_3d
        self.X, self.Y, self.Z = X, Y, Z

        self.vox_util = bev_utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)

        grid_conf = { # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX-XMIN)/float(X)],
            'ybound': [ZMIN, ZMAX, (ZMAX-ZMIN)/float(Z)],
            'zbound': [YMIN, YMAX, (YMAX-YMIN)/float(Y)],
        }
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print(self)


    def __len__(self):
        return len(self.ixes)
    
    
    def get_scenes(self):

        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]
        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples
    
    

    def __getitem__(self, index):

        cams = self.choose_cams()
        
        if self.is_train and self.do_shuffle_cams:
            # randomly sample the ref cam
            refcam_id = np.random.randint(1, len(cams))
        else:
            refcam_id = self.refcam_id

        rec = self.ixes[index]
        imgs, rots, trans, intrins = self.get_image_data(rec, cams)

        imgs = move_refcam(imgs, refcam_id)
        rots = move_refcam(rots, refcam_id)
        trans = move_refcam(trans, refcam_id)
        intrins = move_refcam(intrins, refcam_id)

        lrtlist_, boxlist_, vislist_, tidlist_ = self.get_lrtlist(rec)
        N_ = lrtlist_.shape[0]

        # import ipdb; ipdb.set_trace()
        if N_ > 0:
            velo_T_cam = bev_utils.geom.merge_rt(rots, trans)
            cam_T_velo = bev_utils.geom.safe_inverse(velo_T_cam)
            # note we index 0:1, since we already put refcam into zeroth position
            lrtlist_cam = bev_utils.geom.apply_4x4_to_lrt(cam_T_velo[0:1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)
            seg_bev, valid_bev = self.get_seg_bev(lrtlist_cam, vislist_)
            center_bev, offset_bev = self.get_center_and_offset_bev(lrtlist_cam, seg_bev)[:2]
        else:
            seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)
            center_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            offset_bev = torch.zeros((2, self.Z, self.X), dtype=torch.float32)

        seg_bev = (seg_bev > 0).float()

        return imgs, rots, trans, intrins, seg_bev, valid_bev, center_bev, offset_bev


    def sample_augmentation(self):

        fH, fW = self.data_aug_conf['final_dim']

        if self.is_train:
            
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW*resize), int(fH*resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH)/2)
            crop_w = int((newW - fW)/2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        else: # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        return resize_dims, crop



    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []

        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.dataroot, samp['filename'])       
            img = Image.open(imgname)
            
            W, H = img.size

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            resize_dims, crop = self.sample_augmentation()

            sx = resize_dims[0]/float(W)
            sy = resize_dims[1]/float(H)

            intrin = bev_utils.geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)
            fx, fy, x0, y0 = bev_utils.geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = bev_utils.geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)
            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),torch.stack(intrins))
    

    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = bev_utils.geom.get_xyzlist_from_lrtlist(lrtlist_cam) # B, N, 8, 3
        y_cam = corners_cam[:,:,:,1] # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), self.Z, self.Y, self.X).reshape(B, N, 8, 3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3) # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four

        for n in range(N):
            _, inds = torch.topk(y_cam[0,n], 4, largest=False)
            pts = corners_mem[0,n,inds].numpy().astype(np.int32) # 4, 2

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0],pts[1],pts[3],pts[2]])
            
            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(seg, [pts], n+1.0)
            
            if vislist[n]==0:
                # draw a black rectangle if it's invisible
                cv2.fillPoly(val, [pts], 0.0)

        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0) # 1, Z, X
    

    def get_center_and_offset_bev(self, lrtlist_cam, seg_bev):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        clist_cam = bev_utils.geom.get_clist_from_lrtlist(lrtlist_cam)

        radius = 3
        center, offset = self.vox_util.xyz2circles_bev(clist_cam, radius, self.Z, self.Y, self.X, already_mem=False, also_offset=True)

        masklist = torch.zeros((1, N, 1, self.Z, 1, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            masklist[0,n,0,:,0] = (inst.squeeze() > 0.01).float()

        offset = offset * masklist
        offset = torch.sum(offset, dim=1) # B,3,Z,Y,X

        min_offset = torch.min(offset, dim=3)[0] # B,2,Z,X
        max_offset = torch.max(offset, dim=3)[0] # B,2,Z,X
        offset = min_offset + max_offset
        
        center = torch.max(center, dim=1, keepdim=True)[0] # B,1,Z,Y,X
        center = torch.max(center, dim=3)[0] # max along Y; 1,Z,X
        
        return center.squeeze(0), offset.squeeze(0) 
    

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0)) # invisible
            else:
                vislist.append(torch.tensor(1.0)) # visible
                
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1],l[0],l[2]])
            lrt = bev_utils.py.merge_lrt(l, bev_utils.py.merge_rt(r,t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            # print('rx, ry, rz', rx, ry, rz)
            rs = np.stack([ry*0, ry, ry*0])
            box_ = torch.from_numpy(np.stack([t,l,rs])).reshape(9)
            # print('box_', box_)
            boxlist.append(box_)
        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
            # tidlist = torch.stack(tidlist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros((0))
            # tidlist = torch.zeros((0))
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist
    

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)



class NuScenesDatasetWrapper:
    def __init__(self, args, nusc=None):
        self.args = args

        self.args = args

        if nusc is not None:
            self.nusc = nusc
        else:
            self.nusc = NuScenes(
                                version='v1.0-{}'.format(args.version),
                                dataroot=args.dataset_path,
                                verbose=True
                                )

        final_dim = args.resolution
        
        if args.rand_crop_and_resize:
            resize_lim = [0.8,1.2]
            crop_offset = int(final_dim[0]*(1-resize_lim[0]))
        else:
            resize_lim = [1.0,1.0]
            crop_offset = 0
    
        self.do_shuffle_cams = args.do_shuffle_cams
        self.data_aug_conf = {
            'crop_offset': crop_offset,
            'resize_lim': resize_lim,
            'final_dim': final_dim,
            'H': 900, 'W': 1600,
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'ncams': args.ncams,
        }

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        self.scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(self.scene_centroid_py).float()

        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        YMIN, YMAX = -5, 5
        self.bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

        self.Z, self.Y, self.X = 200, 8, 200


    def train(self):

        traindata = NuscData(
                                self.nusc, 
                                is_train=True, 
                                data_aug_conf=self.data_aug_conf,
                                centroid=self.scene_centroid_py,
                                bounds=self.bounds,
                                res_3d=(self.Z,self.Y,self.X),
                                do_shuffle_cams=self.do_shuffle_cams,
                            )
        
        return traindata
    

    def val(self):

        valdata = NuscData(
                            self.nusc,
                            is_train=False,
                            data_aug_conf=self.data_aug_conf,
                            centroid=self.scene_centroid_py,
                            bounds=self.bounds,
                            res_3d=(self.Z,self.Y,self.X),
                            do_shuffle_cams=False,
                            )
        
        return valdata

  