import cv2

import raytracing
import open3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.database import parse_database_name, get_database_split, BaseDatabase
from network.other_field import SingleVarianceNetwork, NeRFNetwork, TVLoss
from network.fields import *
from utils.network_utils import get_intersection, sample_pdf, extract_geometry, safe_l2_normalize
from utils.base_utils import color_map_forward, downsample_gaussian_blur
from utils.raw_utils import linear_to_srgb
import time

def build_imgs_info(database:BaseDatabase, img_ids, apply_mask_loss=False):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]

    images = np.stack(images, 0)
    images = color_map_forward(images).astype(np.float32)
    Ks = np.stack(Ks, 0).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)
    imgs_info = {
        'imgs': images, 
        'Ks': Ks, 
        'poses': poses,
    }

    if apply_mask_loss:
        masks = [database.get_depth(img_id)[1] for img_id in img_ids]
        masks = np.stack(masks, 0)
        imgs_info['masks'] = masks
    
    return imgs_info

def imgs_info_to_torch(imgs_info, device='cpu'):
    for k, v in imgs_info.items():
        v = torch.from_numpy(v)
        if k.startswith('imgs'): v = v.permute(0,3,1,2)
        imgs_info[k] = v.to(device)
    return imgs_info

def imgs_info_slice(imgs_info, idxs):
    new_imgs_info={}
    for k, v in imgs_info.items():
        new_imgs_info[k]=v[idxs]
    return new_imgs_info

def imgs_info_to_cuda(imgs_info):
    for k, v in imgs_info.items():
        imgs_info[k]=v.cuda()
    return imgs_info

def imgs_info_downsample(imgs_info, ratio):
    b, _, h, w = imgs_info['imgs'].shape
    dh, dw = int(ratio*h), int(ratio*w)
    imgs_info_copy = {k:v for k,v in imgs_info.items()}
    imgs_info_copy['imgs'], imgs_info_copy['Ks'] = [], []
    for bi in range(b):
        img = imgs_info['imgs'][bi].cpu().numpy().transpose([1,2,0])
        img = downsample_gaussian_blur(img, ratio)
        img = cv2.resize(img, (dw,dh), interpolation=cv2.INTER_LINEAR)
        imgs_info_copy['imgs'].append(torch.from_numpy(img).permute(2,0,1))
        K = torch.from_numpy(np.diag([dw / w, dh / h, 1]).astype(np.float32)) @ imgs_info['Ks'][bi]
        imgs_info_copy['Ks'].append(K)

    imgs_info_copy['imgs'] = torch.stack(imgs_info_copy['imgs'], 0)
    imgs_info_copy['Ks'] = torch.stack(imgs_info_copy['Ks'], 0)
    return imgs_info_copy


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

 
class ShapeRenderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        'std_net': 'default',
        'std_act': 'exp',
        'inv_s_init': 0.3,
        'freeze_inv_s_step': None,

        # geometry network
        'sdf_net': 'default',
        'sdf_activation': 'none',
        'sdf_bias': 0.5,
        'sdf_n_layers': 8,
        'sdf_freq': 6,
        'sdf_d_out': 129,
        'geometry_init': True,

        # shader network
        'shader_config': {},

        # sampling strategy
        'n_samples': 64,
        'n_bg_samples': 32,
        'inf_far': 1000.0,
        'n_importance': 64,
        'up_sample_steps': 4,  # 1 for simple coarse-to-fine sampling
        'perturb': 1.0,
        'anneal_end': 50000,
        'train_ray_num': 1024,
        'test_ray_num': 2048,
        'clip_sample_variance': True,

        # dataset
        'database_name': 'nerf_synthetic/lego/black_800',

        # validation
        'test_downsample_ratio': True,
        'downsample_ratio': 0.25,
        'val_geometry': False,

        # losses
        'rgb_loss': 'charbonier',
        'apply_occ_loss': True,
        'apply_tv_loss' : True,
        'apply_sparse_loss' : True,
        'apply_hessian_loss': True,
        'apply_gaussian_loss': False,
        'occ_loss_step': 20000,
        'occ_loss_max_pn': 2048,
        'occ_sdf_thresh': 0.01,
        'apply_gaussian_loss': False,
        'gaussianLoss_step': 20000,

        "fixed_camera": False,
        
        # Tenso
        'device' : 'cuda',
        'gridSize' : [512, 512, 512],
        'aabb' : [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        'step_ratio' : 0.5,
        'alphaMask_thres' : 0.0001,
        'marched_weights_thres' : 0.0001,
        'sdf_n_comp' : 16,
        'app_n_comp' : 36,
        'sdf_dim' : 128,
        'app_dim' : 128,  
        'max_levels': 1,
        'has_radiance_field': False,
        'radiance_field_step': 0,
        'predict_BG': True,
        'isBGWhite': True,

        # dataset
        'nerfDataType': False,
        'split_manul': False,
        'apply_mask_loss': False,

        # alphaMask multi Length
        'mul_length': 10,
    }

    def __init__(self, cfg, training=True):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        
        self.device = self.cfg['device']
        gridSize = torch.tensor(self.cfg['gridSize'])
        max_levels = self.cfg['max_levels']
        self.aabb = torch.tensor(self.cfg['aabb'], device=self.device)
        self.center = torch.mean(self.aabb, axis=0).float().view(1, 1, 3)
        self.radius = (self.aabb[1] - self.center).mean().float()
        self.alphaMask = None
        self.step_ratio = self.cfg['step_ratio']
        self.alphaMask_thres = self.cfg['alphaMask_thres']
        self.marched_weights_thres = self.cfg['marched_weights_thres']
        self.sdf_n_comp, self.app_n_comp = self.cfg['sdf_n_comp'], self.cfg['app_n_comp']
        self.sdf_dim, self.app_dim = self.cfg['sdf_dim'], self.cfg['app_dim']
        self.update_stepSize(gridSize, max_levels)
        
        self.sdf_network = TensoSDF(
            self.gridSize, self.aabb, device=self.device, init_n_levels=self.max_levels,
            sdf_n_comp=self.sdf_n_comp, sdf_dim=self.sdf_dim, app_dim=self.app_dim)

        self.tv_reg = TVLoss()
        self.deviation_network = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act'])

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        if self.cfg['predict_BG']:
            self.outer_nerf = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4, skips=[4], use_viewdirs=True)
            nn.init.constant_(self.outer_nerf.rgb_linear.bias, np.log(0.5))
        else:
            self.cfg['n_bg_samples'] = 0

        self.cfg['shader_config'] = {
            'occ_loss_step': self.cfg['occ_loss_step'],
            'has_radiance_field': self.cfg['has_radiance_field'],
            'radiance_field_step': self.cfg['radiance_field_step'],
        }
        
        self.color_network = ShapeShadingNetwork(self.cfg['shader_config'])
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x, None)

        if training:
            self._init_dataset()
        
    def update_stepSize(self, gridSize, max_levels):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)        
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = torch.tensor(gridSize.cpu(), dtype=torch.int32).to(self.device)
        self.max_levels = max_levels
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units)*self.step_ratio
        print("sampling step size: ", self.stepSize)
    
    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(128,128,128)):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        alpha, grid_xyz = self.compute_gridAlpha(gridSize)
        grid_xyz = grid_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None] # (1,1,gridSize012,)
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = grid_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))

        torch.set_default_tensor_type('torch.FloatTensor')
        return new_aabb

    @torch.no_grad()    
    def compute_gridAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else torch.LongTensor(gridSize).to(self.device)
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        grid_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        stepLength = torch.mean(self.aabbSize / (gridSize - 1))
        alpha = torch.zeros_like(grid_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_grid_alpha(grid_xyz[i].view(-1,3), stepLength).view((gridSize[1], gridSize[2]))
        return alpha, grid_xyz

    def compute_grid_alpha(self, xyz_locs, length):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        
        alpha = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = xyz_locs[alpha_mask]
            sdfs = self.sdf_inter_fun(xyz_sampled)[..., 0]
            near_surf_mask = torch.abs(sdfs) < self.cfg['mul_length'] * length
            inv_s = self.deviation_network(xyz_sampled).clip(1e-6, 1e6)
            inv_s = inv_s[..., 0]
            estimated_next_sdf = sdfs - length * 0.5
            estimated_prev_sdf = sdfs + length * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)       # [N_rays, ]
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha_weights = ((p + 1e-5) / (c + 1e-5)).clip(min=0.0, max=1.0)
            alpha_weights[near_surf_mask] = 1
            alpha[alpha_mask] = alpha_weights
        return alpha      

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'sdf_n_comp': self.sdf_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'sdf_dim': self.sdf_dim,
            'app_dim': self.app_dim,

            'alphaMask_thres': self.alphaMask_thres,
            'marched_weights_thres' : self.marched_weights_thres,
            'step_ratio': self.step_ratio,
            'max_levels': self.max_levels,
        }

    def ckpt_to_save(self):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'network_state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        return ckpt

    def load_ckpt(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['network_state_dict'])

    def upsample_sdf_grid(self, res_target):
        res_target = torch.tensor(res_target, device=self.device)
        new_res, max_levels = self.sdf_network.upsample_volume_grid(res_target)
        self.update_stepSize(new_res, max_levels)

    def shrink_sdf_grid(self, new_aabb):
        raise NotImplementedError

    def get_train_opt_params(self, learning_rate_xyz, learning_rate_net):
        grad_vars = []
        get_grad_vars_from_net = lambda net : [{'params' : net.parameters(), 'lr' : learning_rate_net}]
        grad_vars += self.sdf_network.get_optparam_groups(learning_rate_xyz, learning_rate_net)
        grad_vars += get_grad_vars_from_net(self.deviation_network)
        grad_vars += get_grad_vars_from_net(self.color_network)
        if self.cfg['predict_BG']:
            grad_vars += get_grad_vars_from_net(self.outer_nerf)
        return grad_vars
        
    def _init_dataset(self):
        # train/test split
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'], isWhiteBG=self.cfg['isBGWhite'])
        self.train_ids, self.test_ids = get_database_split(self.database, split_manul=self.cfg['split_manul'])
        self.train_ids = np.asarray(self.train_ids)

        self.train_imgs_info = build_imgs_info(self.database, self.train_ids, apply_mask_loss=self.cfg['apply_mask_loss'])
        self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
        b, _, h, w = self.train_imgs_info['imgs'].shape
        print(f'training size {h} {w} ...')
        self.train_num = len(self.train_ids)

        self.test_imgs_info = build_imgs_info(self.database, self.test_ids, apply_mask_loss=self.cfg['apply_mask_loss'])
        self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
        self.test_num = len(self.test_ids)
        print(f'Acutal splits num: train -> {self.train_num}, val -> {self.test_num}')
        # clean the data if we already have
        if hasattr(self, 'train_batch'):
            del self.train_batch

        if self.cfg['nerfDataType']:
            self.train_batch, _, _, _ = self._construct_ray_batch_nerf(self.train_imgs_info)        
        else:    
            self.train_batch, _, _, _ = self._construct_ray_batch(self.train_imgs_info, apply_mask=self.cfg['apply_mask_loss'])
        
        self.filtering_train_rays()
        self._shuffle_train_batch()

    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.tbn, device='cpu')  # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def _construct_ray_batch(self, imgs_info, device='cpu', apply_mask=False):
        imn, _, h, w = imgs_info['imgs'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(imn, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(imn, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(imn, h * w, 1, dtype=torch.float32, device=device)], 2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        dirs = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1) # imn,h*w,3
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4

        rn = imn * h * w
        dirs = dirs.float().reshape(rn, 3).to(device)
        imgs = imgs.float().reshape(rn, 3).to(device)
        idxs = idxs.long().reshape(rn, 1).to(device)

        idxs = idxs[..., 0]  # rn
        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        dirs = poses[idxs, :, :3].permute(0, 2, 1) @ dirs.unsqueeze(-1)
        dirs = dirs[..., 0]  # rn,3
        dirs = F.normalize(dirs, dim=-1)

        human_poses = self.get_human_coordinate_poses(poses)[idxs]  # rn, 3, 4
        ray_batch = {
            'dirs': dirs,
            'rays_o': rays_o,
            'rgbs': imgs,
            'human_poses': human_poses,
        }

        if apply_mask:
            masks = imgs_info['masks'].reshape(imn, h * w).float().reshape(rn, 1).to(device)
            ray_batch['masks'] = masks
        return ray_batch, rn, h, w

    def _construct_ray_batch_nerf(self, imgs_info, device='cpu', is_train=True):
        imn, _, h, w = imgs_info['imgs'].shape
        i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i-K[0][2]+0.5)/K[0][0], -(j-K[1][2]+0.5)/K[1][1], -torch.ones_like(i)], -1) # h, w, 3
        dirs = dirs[None, ...].repeat(imn, 1, 1, 1) # imn, h, w, 3
        
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn, 4, 4
        rays_o = poses[..., :3, -1][:, None, :].repeat(1, h*w, 1) # imn, h*w, 3
        
        rn = imn * h * w
        dirs = dirs.float().reshape(rn, 3).to(device) # rn,3
        idxs = idxs.long().reshape(rn, 1).to(device)  # rn,1 
        rays_o = rays_o.float().reshape(rn, 3).to(device) # rn,3
        imgs = imgs.float().reshape(rn, 3).to(device) # rn,3
        
        dirs = torch.sum(dirs[..., None, :] * poses[idxs[..., 0], :3, :3], -1)  # rn,3
        dirs = F.normalize(dirs, dim=-1)
        
        ray_batch = {
            'dirs': dirs,
            'rays_o': rays_o,
            'rgbs': imgs,
            'human_poses': poses[idxs[..., 0], :3, :],
        }

        if is_train:
            masks = imgs_info['masks'].reshape(imn, h * w).float().reshape(rn, 1).to(device)
            ray_batch['masks'] = masks
        return ray_batch, rn, h, w

    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3], device=poses.device).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3]).to(poses.device)  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    @torch.no_grad()
    def filtering_train_rays(self, device='cuda', chunk=10240*5):
        print('========> filtering rays ...')
        tt = time.time()
        rays_o, rays_d = self.train_batch['rays_o'], self.train_batch['dirs']  
        N = torch.tensor(rays_o.shape[:-1]).cpu().prod()
        aabb = self.aabb.to(device)

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)        
        for idx_chunk in idx_chunks:
            rays_o_chunk, rays_d_chunk = rays_o[idx_chunk].to(device), rays_d[idx_chunk].to(device)

            vec = torch.where(rays_d_chunk == 0, torch.full_like(rays_d_chunk, 1e-6), rays_d_chunk)
            rate_a = (aabb[1] - rays_o_chunk) / vec
            rate_b = (aabb[0] - rays_o_chunk) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min
            
            mask_filtered.append(mask_inbbox.cpu())
            
        mask_filtered = torch.cat(mask_filtered).view(rays_o.shape[:-1])
        valid_rn = torch.sum(mask_filtered)
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {valid_rn / N}')
        
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[mask_filtered]
        self.tbn = valid_rn

    @torch.no_grad()     
    def nvs(self, pose, K, h, w):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)
        rn = h * w

        def construct_ray_dirs():
            coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
            coords = coords.to(device)
            coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
            coords = coords.reshape(1, h * w, 2)
            coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3
            # 1,h*w,3 @ imn,3,3 => 1,h*w,3
            dirs = coords @ torch.inverse(K).permute(0, 2, 1)
            dirs = dirs.reshape(-1, 3)
            dirs = F.normalize(dirs, dim=-1)         
            ray_batch = {
                'dirs': dirs.float().to(device),
            }
            return ray_batch
        
        def construct_ray_dirs_nerf():
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
            i = i.t().to(device)
            j = j.t().to(device)
            k = K[0]
            dirs = torch.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -torch.ones_like(i).to(device)], -1) # h, w, 3
            dirs = F.normalize(dirs, dim=-1)
            dirs = dirs.reshape(-1, 3)

            idxs = torch.arange(1, dtype=torch.int64, device=device)[:, None, None].repeat(1, rn, 1)  # 1,rn,1
            idxs = idxs.long().reshape(rn, 1).to(device)
            
            ray_batch = {
                'dirs': dirs.float().to(device),
                'human_poses': pose[idxs[..., 0], :3, :],
                'idxs': idxs,
            }
            return ray_batch
        
        if self.cfg['nerfDataType']:
            ray_batch = construct_ray_dirs_nerf()
            process_ray_batch = self._process_ray_batch_nerf
        else:
            ray_batch = construct_ray_dirs()
            process_ray_batch = self._process_ray_batch            

        trn = 2048
        output = {
                  'color' : [], 'albedo' : [], 'roughness' : [], 'normal' : [], 'normal_vis': [],
                  'occ_predict' : [], 'occ_trace' : [], 
                  'diff_color' : [], 'spec_color' : [], 
                  'diff_light' : [], 'spec_light' : [],
                  'indirect_light' : [], 
                  }
        if self.cfg['has_radiance_field']:
            output['radiance'] = []
        for ri in range(0, rn, trn):
            cur_ray_batch = {}
            for k, v in ray_batch.items(): cur_ray_batch[k] = v[ri:ri + trn]

            with torch.no_grad():
                cur_ray_batch, near, far = process_ray_batch(cur_ray_batch, pose)
                human_poses = cur_ray_batch['human_poses']
                cur_outputs = self.render(cur_ray_batch, near, far, human_poses, is_train=False, step=300000)
                output['color'].append(cur_outputs['ray_rgb'].detach().cpu().numpy())
                output['albedo'].append(cur_outputs['albedo'].detach().cpu().numpy())
                output['roughness'].append(cur_outputs['roughness'].detach().cpu().numpy())
                output['normal'].append(cur_outputs['normal'].detach().cpu().numpy())
                output['normal_vis'].append(cur_outputs['normal_vis'].detach().cpu().numpy())

                output['occ_predict'].append(cur_outputs['occ_prob'].detach().cpu().numpy())
                output['occ_trace'].append(cur_outputs['occ_prob_gt'].detach().cpu().numpy())
                
                output['diff_color'].append(cur_outputs['diffuse_color'].detach().cpu().numpy())
                output['spec_color'].append(cur_outputs['specular_color'].detach().cpu().numpy())
                
                output['diff_light'].append(cur_outputs['diffuse_light'].detach().cpu().numpy())
                output['spec_light'].append(cur_outputs['specular_light'].detach().cpu().numpy())
                output['indirect_light'].append(cur_outputs['indirect_light'].detach().cpu().numpy())
                if self.cfg['has_radiance_field']:
                    output['radiance'].append(cur_outputs['radiance'].detach().cpu().numpy())
        for k in output:        
            val = np.concatenate(output[k], 0)
            output[k] = np.reshape(val, [h, w, val.shape[-1]])
        torch.set_default_tensor_type('torch.FloatTensor')
        return output

    def get_anneal_val(self, step):
        if self.cfg['anneal_end'] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg['anneal_end']])

    def near_far_from_sphere(self, rays_o, rays_d):
        radius = self.radius if self.radius is not None else 1.0
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - radius
        far = mid + radius
        near = torch.clamp(near, min=1e-3)
        return near, far

    def compute_sample_level(self, pts):
        level = torch.zeros(pts.shape[:-1] + (1, ))
        return level

    def _process_ray_batch(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
        rays_d = rays_d[..., 0]  # rn,3

        rays_o = rays_o
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far  # rn, 3, 4
    
    def _process_ray_batch_nerf(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[idxs, :3, -1] # rn,3
        rays_d = torch.sum(rays_d[..., None, :] * poses[idxs, :3, :3], -1)  # rn,3
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        ray_batch['rays_o'] = rays_o
        ray_batch['dirs'] = rays_d
        return ray_batch, near, far # rn, 3, 4

    def test_step(self, index, step,):
        target_imgs_info, target_img_ids = self.test_imgs_info, self.test_ids
        imgs_info = imgs_info_slice(target_imgs_info, torch.from_numpy(np.asarray([index], np.int64)))
        gt_depth, gt_mask = self.database.get_depth(target_img_ids[index])  # used in evaluation
        if self.cfg['test_downsample_ratio']:
            imgs_info = imgs_info_downsample(imgs_info, self.cfg['downsample_ratio'])
            h, w = gt_depth.shape
            dh, dw = int(self.cfg['downsample_ratio'] * h), int(self.cfg['downsample_ratio'] * w)
            gt_depth, gt_mask = cv2.resize(gt_depth, (dw, dh), interpolation=cv2.INTER_NEAREST), \
                cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt_depth, gt_mask = torch.from_numpy(gt_depth), torch.from_numpy(gt_mask.astype(np.int32))
        if self.cfg['nerfDataType']:
            ray_batch, rn, h, w = self._construct_ray_batch_nerf(imgs_info, is_train=False)        
        else:        
            ray_batch, rn, h, w = self._construct_ray_batch(imgs_info)

        for k, v in ray_batch.items(): ray_batch[k] = v.cuda()

        trn = self.cfg['test_ray_num']
        outputs_keys = ['ray_rgb', 'gradient_error', 'depth', 'acc', 'normal_vis']
        outputs_keys += [
                'diffuse_albedo', 'diffuse_light', 'diffuse_color',
                'specular_albedo', 'specular_light', 'specular_color', 'specular_ref', 'specular_direct_light',
                'metallic', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
            ]
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            outputs_keys.append('radiance')
            outputs_keys.append('roughness_weights')
        if self.color_network.cfg['human_light']:
            outputs_keys += ['human_light']
        outputs = {k: [] for k in outputs_keys}
        
        for ri in range(0, rn, trn):
            cur_ray_batch= {k:v[ri:ri+trn] for k, v in ray_batch.items()}
            rays_o, rays_d, human_poses = cur_ray_batch['rays_o'], cur_ray_batch['dirs'], cur_ray_batch['human_poses']
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            
            cur_outputs = self.render(cur_ray_batch, near, far, human_poses, 0, 0, is_train=False, step=step)
            for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

        for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], ray_batch['rgbs'])
        outputs['gt_rgb'] = ray_batch['rgbs'].reshape(h, w, 3)
        outputs['ray_rgb'] = outputs['ray_rgb'].reshape(h, w, 3)
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            outputs['loss_radiance'] = self.compute_rgb_loss(outputs['radiance'], ray_batch['rgbs']) * outputs['roughness_weights']
            outputs['loss_rgb'] = outputs['loss_rgb'] * (1.0 - outputs['roughness_weights'])
            outputs['radiance'] = outputs['radiance'].reshape(h, w, 3)

        # used in evaluation
        outputs['gt_depth'] = gt_depth.unsqueeze(-1)
        outputs['gt_mask'] = gt_mask.unsqueeze(-1)

        self.zero_grad()
        return outputs

    def train_step(self, step):
        rn = self.cfg['train_ray_num']
        # fetch to gpu
        train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + rn].cuda() for k, v in self.train_batch.items()}
        self.train_batch_i += rn
        if self.train_batch_i + rn >= self.tbn: self._shuffle_train_batch()
        rays_o, rays_d, human_poses = train_ray_batch['rays_o'], train_ray_batch['dirs'], train_ray_batch['human_poses']
        near, far = self.near_far_from_sphere(rays_o, rays_d)
        outputs = self.render(train_ray_batch, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True, step=step)

        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            outputs['loss_radiance'] = self.compute_rgb_loss(outputs['radiance'], train_ray_batch['rgbs']) * outputs['roughness_weights']  # ray_loss
            outputs['loss_rgb'] = outputs['loss_rgb'] * (1.0 - outputs['roughness_weights'])
        if self.cfg['apply_mask_loss']:
            outputs['loss_mask'] = F.binary_cross_entropy(outputs['acc'].clip(1e-3, 1.0 - 1e-3), (train_ray_batch['masks'] > 0.5).float())
        return outputs

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l2':
            rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
        elif self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'smooth_l1':
            rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    def compute_density(self, points):
        points_norm = torch.norm(points, dim=-1, keepdim=True)
        points_norm = torch.clamp(points_norm, min=1e-3)
        sigma = self.outer_nerf.density(torch.cat([points / points_norm, 1.0 / points_norm], -1))[..., 0]
        return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        level = self.compute_sample_level(pts) # [rn, sn, 1]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), level).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def sample_ray(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.cfg['n_samples'] 
        n_bg_samples = self.cfg['n_bg_samples']
        n_importance = self.cfg['n_importance']
        up_sample_steps = self.cfg['up_sample_steps']

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near[..., 0], max=far[..., 0]).unsqueeze(-1)
        
        z_vals = t_min + (t_max - t_min) * z_vals[None, :]  # rn,sn
        if n_bg_samples > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples)

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            if n_bg_samples > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if n_bg_samples > 0:
            z_vals_outside = t_max / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [rn, sn, 3]
            level = self.compute_sample_level(pts) # [rn, sn, 1]
            sdf = self.sdf_network.sdf(pts, level).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg['clip_sample_variance']:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        if n_bg_samples > 0:
            z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals

    def render(self, ray_batch, near, far, human_poses, perturb_overwrite=-1, cos_anneal_ratio=0.0, is_train=True, step=None):
        """
        :param ray_batch: rn,x
        :param near:   rn,1
        :param far:    rn,1
        :param human_poses:     rn,3,4
        :param perturb_overwrite: set 0 for inference
        :param cos_anneal_ratio:
        :param is_train:
        :param step:
        :return:
        """
        perturb = self.cfg['perturb']
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['dirs']
        z_vals = self.sample_ray(rays_o, rays_d, near, far, perturb)
        ret = self.render_core(rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=cos_anneal_ratio, step=step, is_train=is_train)
        return ret

    def compute_validation_info(self, z_vals, rays_o, rays_d, weights, human_poses, step):
        depth = torch.sum(weights * z_vals, -1, keepdim=True)  # rn, 1
        points = depth * rays_d + rays_o  # rn,3
        level = self.compute_sample_level(points) # [rn, 1]
        gradients, _ = self.sdf_network.gradient(points, level, training=False)  # rn,3
        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask[..., None]

        outputs = {
            'depth': depth,  # rn,1
        }

        if not self.cfg['nerfDataType']:
            outputs['normal_vis'] = ((F.normalize(gradients, dim=-1) + 1.0) * 0.5) * inner_mask

        feature_vector = self.sdf_network(points, level)[..., 1:]  # rn,f
        _, occ_info, inter_results = self.color_network(points, gradients, -F.normalize(rays_d, dim=-1), feature_vector, human_poses, inter_results=True, step=step)
        _, occ_prob, _ = get_intersection(self.sdf_inter_fun, self.deviation_network, points, occ_info['reflective'], sn0=128, sn1=9)  # pn,sn-1
        occ_prob_gt = torch.sum(occ_prob, dim=-1, keepdim=True)
        outputs['occ_prob_gt'] = occ_prob_gt
        for k, v in inter_results.items(): inter_results[k] = v * inner_mask
        outputs.update(inter_results)
        return outputs 
        
    def compute_sdf_alpha(self, points, level, dists, dirs, cos_anneal_ratio, step, is_train):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network(points, level)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients, hessian = self.sdf_network.gradient(points, level, training=is_train, sdf=sdf[..., None])  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg['freeze_inv_s_step'] is not None and step < self.cfg['freeze_inv_s_step']:
            inv_s = inv_s.detach()

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf, hessian

    def compute_density_alpha(self, points, dists, dirs, nerf):
        norm = torch.norm(points, dim=-1, keepdim=True)
        points = torch.cat([points / norm, 1.0 / norm], -1)
        density, color = nerf(points, dirs)  # [...,1] [...,3]
        alpha = self.density_activation(density[..., 0], dists)
        color = linear_to_srgb(torch.exp(torch.clamp(color, max=5.0)))
        return alpha, color

    def compute_occ_loss(self, occ_info, points, sdf, gradients, dirs, step):
        if step < self.cfg['occ_loss_step']: return torch.zeros(1)

        occ_prob = occ_info['occ_prob']
        reflective = occ_info['reflective']

        # select a subset for occ loss
        # note we only apply occ loss on the surface
        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask
        
        sdf_mask = torch.abs(sdf) < self.cfg['occ_sdf_thresh']
        normal_mask = torch.sum(gradients * dirs, -1) < 0  # pn
        mask = (inner_mask & normal_mask & sdf_mask)

        if torch.sum(mask) > self.cfg['occ_loss_max_pn']:
            indices = torch.nonzero(mask)[:, 0]  # npn
            idx = torch.randperm(indices.shape[0], device='cuda')  # npn
            indices = indices[idx[:self.cfg['occ_loss_max_pn']]]  # max_pn
            mask_new = torch.zeros_like(mask)
            mask_new[indices] = 1
            mask = mask_new

        if torch.sum(mask) > 0:
            inter_dist, inter_prob, inter_sdf = get_intersection(self.sdf_inter_fun, self.deviation_network, points[mask], reflective[mask], sn0=64, sn1=16)  # pn,sn-1
            occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
            return F.l1_loss(occ_prob[mask], occ_prob_gt)
        else:
            return torch.zeros(1)

    def render_core(self, rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=0.0, step=None, is_train=True):
        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5
        
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1) # [rn, sn, 3]
        level = self.compute_sample_level(points) # [rn, sn, 1]
        derived_normals = torch.zeros(batch_size, n_samples, 3)

        outer_mask = ((self.aabb[0]>points) | (points>self.aabb[1])).any(dim=-1)
        inner_mask = ~outer_mask      

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        human_poses_pt = human_poses.unsqueeze(-3).expand(batch_size, n_samples, 3, 4)
        dirs = F.normalize(dirs, dim=-1)
        alpha, sampled_color = torch.zeros(batch_size, n_samples), torch.zeros(batch_size, n_samples, 3)
        if self.cfg['predict_BG'] and torch.sum(outer_mask) > 0:
            alpha[outer_mask], sampled_color[outer_mask] = self.compute_density_alpha(points[outer_mask], dists[outer_mask], -dirs[outer_mask], self.outer_nerf)
        
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            sampled_radiance = torch.zeros(batch_size, n_samples, 3)
            roughness = torch.zeros(batch_size, n_samples, 1)

        alpha_rest_ratio = 1.0
        if self.alphaMask is not None:
            alpha_mask = self.alphaMask.sample_alpha(points[inner_mask]) > 0
            alpha_rest_ratio = 1.0 - torch.sum(~alpha_mask) / torch.sum(inner_mask)
            inner_mask_invalid = ~inner_mask
            inner_mask_invalid[inner_mask] |= (~alpha_mask)
            inner_mask = ~inner_mask_invalid 
                
        if torch.sum(inner_mask) > 0:                
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf, hessian = self.compute_sdf_alpha(points[inner_mask], level[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step, is_train)
            valid_normals = gradients
            derived_normals[inner_mask] = gradients
            if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
                sampled_color[inner_mask], sampled_radiance[inner_mask], occ_info = self.color_network(points[inner_mask], valid_normals, -dirs[inner_mask], feature_vector, human_poses_pt[inner_mask], step=step)      
                roughness[inner_mask] = occ_info['roughness']      
            else:
                sampled_color[inner_mask], _, occ_info = self.color_network(points[inner_mask], valid_normals, -dirs[inner_mask], feature_vector, human_poses_pt[inner_mask], step=step)
            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
            
            if self.cfg['apply_sparse_loss']:
                gamma = 20.
                reg_loss = torch.exp(-gamma * sdf.abs())        # [..., ]
                reg_loss = reg_loss.sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
                
            if self.cfg['apply_hessian_loss'] and hessian is not None:
                hessian_loss = hessian.abs().sum() / (inner_mask.sum() + 1e-5) * alpha_rest_ratio
            else:
                hessian_loss = torch.zeros(1)
        else:
            gradient_error = torch.zeros(1)
            if self.cfg['apply_sparse_loss']:
                reg_loss = torch.zeros(1)
            if self.cfg['apply_hessian_loss']:
                hessian_loss = torch.zeros(1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[..., :-1]  # rn,sn
        acc_map = torch.sum(weights, dim=-1, keepdim=True) # rn, 1          
        if not self.cfg['predict_BG'] and self.cfg['isBGWhite']:
            color = (sampled_color * weights[..., None]).sum(dim=1) + (1 - acc_map)
        else:
            color = (sampled_color * weights[..., None]).sum(dim=1)
        outputs = {
            'ray_rgb': color,  # rn,3
            'gradient_error': gradient_error,  # rn
            'acc' : acc_map,
        }

        acc_sampled_normal = (derived_normals * weights[..., None]).sum(dim=1)
        outputs['normal'] = F.normalize(acc_sampled_normal * acc_map + (1. - acc_map) * torch.tensor([0.0, 0.0, 1.0], device=acc_sampled_normal.device), dim=-1)
        
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            if not self.cfg['predict_BG'] and self.cfg['isBGWhite']:
                radiance = (sampled_radiance * weights[..., None]).sum(dim=1) + (1 - acc_map)
            else:
                radiance = (sampled_radiance * weights[..., None]).sum(dim=1)
            roughness_weights = (roughness * weights[..., None]).sum(dim=1) # rn, 1
            outputs['radiance'] = radiance
            outputs['roughness_weights'] = roughness_weights.squeeze(-1).clone().detach() # rn

        if torch.sum(inner_mask) > 0:
            outputs['std'] = torch.mean(1 / inv_s)
        else:
            outputs['std'] = torch.zeros(1)

        if step < 1000:
            if torch.sum(inner_mask) > 0:
                outputs['sdf_pts'] = points[inner_mask]
                outputs['sdf_vals'] = self.sdf_network.sdf(points[inner_mask], level[inner_mask])[..., 0]
            else:
                outputs['sdf_pts'] = torch.zeros(1)
                outputs['sdf_vals'] = torch.zeros(1)

        if self.cfg['apply_occ_loss']:
            # occlusion loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_occ'] = self.compute_occ_loss(occ_info, points[inner_mask], sdf, valid_normals, dirs[inner_mask], step)
            else:
                outputs['loss_occ'] = torch.zeros(1)

        if self.cfg['apply_gaussian_loss'] and step > self.cfg['gaussianLoss_step']:
            # gaussian loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_gaussian'] = self.sdf_network.grid_gaussian_loss()
            else:
                outputs['loss_gaussian'] = torch.zeros(1)

        if self.cfg['apply_tv_loss']:
            outputs['loss_tv_sdf'] = self.sdf_network.TV_loss_sdf(self.tv_reg)          

        if self.cfg['apply_sparse_loss']:
            outputs['loss_sparse'] = reg_loss

        if self.cfg['apply_hessian_loss']:
            outputs['loss_hessian'] = hessian_loss

        if not is_train:
            outputs['normal_vis'] = ((outputs['normal'] + 1.0) * 0.5) * acc_map + (1. - acc_map)
            outputs.update(self.compute_validation_info(z_vals, rays_o, rays_d, weights, human_poses, step))

        return outputs

    def forward(self, data):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        is_train = 'eval' not in data
        step = data['step']

        if is_train:
            outputs = self.train_step(step)
        else:
            index = data['index']
            outputs = self.test_step(index, step=step)

            if index == 0 and self.cfg['val_geometry']:
                bbox_min = -torch.ones(3)
                bbox_max = torch.ones(3)
                vertices, triangles = extract_geometry(bbox_min, bbox_max, 128, 0, lambda x: self.sdf_network.sdf(x))
                outputs['vertices'] = vertices
                outputs['triangles'] = triangles

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def predict_materials(self):
        name = self.cfg['name']
        mesh = open3d.io.read_triangle_mesh(f'data/meshes/{name}-300000.ply')
        xyz = np.asarray(mesh.vertices)
        xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        metallic, roughness, albedo = [], [], []
        batch_size = 8192
        for vi in range(0, xyz.shape[0], batch_size):
            feature_vectors = self.sdf_network(xyz[vi:vi + batch_size])[:, 1:]
            m, r, a = self.color_network.predict_materials(xyz[vi:vi + batch_size], feature_vectors)
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {'metallic': np.concatenate(metallic, 0),
                'roughness': np.concatenate(roughness, 0),
                'albedo': np.concatenate(albedo, 0)}