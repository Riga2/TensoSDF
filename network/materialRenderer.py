import os
import cv2
import raytracing
import open3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.base_utils import color_map_forward, downsample_gaussian_blur, getHaltonSamples, getStratifiedSamples2D
from dataset.database import parse_database_name, get_database_split, BaseDatabase

from network.fields import MCShadingNetwork
from network.fields import TensoSDF
from network.other_field import SingleVarianceNetwork
from utils.network_utils import get_intersection, get_weights, sample_pdf
import time

def build_imgs_info(database:BaseDatabase, img_ids, is_nerf=False):
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

    if is_nerf:
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

 
class MaterialRenderer(nn.Module):
    default_cfg={
        'train_ray_num': 512,
        'test_ray_num': 512,

        'database_name': 'real/bear/raw_1024',
        'rgb_loss': 'charbonier',

        'mesh': 'data/meshes/bear_shape-300000.ply',

        'shader_cfg': {},

        'reg_mat': True,
        'reg_diffuse_light': True,
        'reg_diffuse_light_lambda': 0.1,
        'fixed_camera': False,

        # dataset
        'nerfDataType': False,
        'split_manul': False,
        
        # geo
        'device': 'cuda',
        'direct_sn0': 128,
        'direct_sn1': 9,
        'sec_sn0': 64,
        'sec_sn1': 6,
        
        # model path
        'geo_model_path': '',
        'std_act': 'exp',
        'inv_s_init': 0.3,
    }
    def __init__(self, cfg, training=True, nvs=False):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()
        self.warned_normal = False
        self.device = self.cfg['device']
        self._init_geometry()
        if os.path.exists(self.cfg['geo_model_path']):
            checkpoint=torch.load(self.cfg['geo_model_path'])
            self.init_sdf(checkpoint)
        if not nvs:
            self._init_dataset(training)
        self._init_shader()

    def _init_geometry(self):
        self.mesh = open3d.io.read_triangle_mesh(self.cfg['mesh'])
        self.ray_tracer = raytracing.RayTracer(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles))

    def init_sdf(self, ckpt):
        step = ckpt['step']

        configs = ckpt['kwargs']
        self.aabb = configs['aabb']
        self.gridSize = torch.tensor(configs['gridSize'], device=self.device)
        self.center = torch.mean(self.aabb, axis=0).float().view(1, 1, 3)
        self.radius = (self.aabb[1] - self.center).mean().float()
        self.unit_size = torch.mean((self.aabb[1] - self.aabb[0]) / (self.gridSize - 1), dim=-1)
        self.sdf_network = TensoSDF(
            torch.tensor(configs['gridSize'], device=self.device), configs['aabb'], device=self.device, init_n_levels=configs['max_levels'],
            sdf_n_comp=configs['sdf_n_comp'], sdf_dim=configs['sdf_dim'], app_dim=configs['app_dim']).to(self.device)

        self.deviation_net = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act']).to(self.device)
        
        sdf_model_dict = self.sdf_network.state_dict()
        deviation_dict = self.deviation_net.state_dict()
        sdf_trained_dict = {'.'.join(k.split('.')[1:]) : v for k, v in ckpt['network_state_dict'].items() if (k.startswith('sdf'))}
        deviation_trained_dict = {'.'.join(k.split('.')[1:]) : v for k, v in ckpt['network_state_dict'].items() if (k.startswith('deviation'))}
        sdf_model_dict.update(sdf_trained_dict)
        deviation_dict.update(deviation_trained_dict)
        self.sdf_network.load_state_dict(sdf_model_dict)
        self.deviation_net.load_state_dict(deviation_trained_dict)
        for params in self.sdf_network.parameters():
            params.requires_grad = False
        for params in self.deviation_net.parameters():
            params.requires_grad = False
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x, None)
        print(f'Geometry starts from step {step}')

    def _init_dataset(self, training):
        # train/test split
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'])
        self.train_ids, self.test_ids = get_database_split(self.database, split_manul=self.cfg['split_manul'])
        self.train_ids = np.asarray(self.train_ids)

        if training:
            self.train_imgs_info = build_imgs_info(self.database, self.train_ids)
            self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
            self.train_num = len(self.train_ids)

            self.test_imgs_info = build_imgs_info(self.database, self.test_ids)
            self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
            self.test_num = len(self.test_ids)
            print(f'Acutal splits num: train -> {self.train_num}, val -> {self.test_num}')

            self.train_ids = torch.arange(len(self.train_ids))
            self.test_ids = torch.arange(len(self.test_ids))
            self._shuffle_train_ids()

    def _init_shader(self):
        self.cfg['shader_cfg']['is_real'] = self.cfg['database_name'].startswith('real')
        self.shader_network = MCShadingNetwork(self.cfg['shader_cfg'], lambda o,d: self.trace(o+2*self.unit_size*d, d))

    def get_train_opt_params(self, learning_rate_xyz, learning_rate_net):
        grad_vars = []
        get_grad_vars_from_net = lambda net : [{'params' : net.parameters(), 'lr' : learning_rate_net}]
        grad_vars += get_grad_vars_from_net(self.shader_network)
        return grad_vars

    def ckpt_to_save(self):
        ckpt = {'network_state_dict': self.state_dict()}
        return ckpt

    def load_ckpt(self, ckpt):
       self.load_state_dict(ckpt['network_state_dict'])
    
    def trace_in_batch(self, rays_o, rays_d, batch_size=512**2, cpu=False):
        inters, normals, depth, hit_mask = [], [], [], []
        rn = rays_o.shape[0]
        for ri in range(0, rn, batch_size):
            inters_cur, normals_cur, depth_cur, hit_mask_cur = self.trace(rays_o[ri:ri+batch_size], rays_d[ri:ri+batch_size])
            if cpu:
                inters_cur = inters_cur.cpu()
                normals_cur = normals_cur.cpu()
                depth_cur = depth_cur.cpu()
                hit_mask_cur = hit_mask_cur.cpu()
            inters.append(inters_cur)
            normals.append(normals_cur)
            depth.append(depth_cur)
            hit_mask.append(hit_mask_cur)
        return torch.cat(inters, 0), torch.cat(normals, 0), torch.cat(depth, 0), torch.cat(hit_mask, 0)

    def trace(self, rays_o, rays_d):
        inters, normals, depth = self.ray_tracer.trace(rays_o, rays_d)
        depth = depth.reshape(*depth.shape, 1)
        normals = -normals
        normals = F.normalize(normals, dim=-1)
        if not self.warned_normal:
            print('warn!!! the normals are flipped in NeuS by default. You may flip the normal according to your mesh!')
            self.warned_normal=True
        miss_mask = depth >= 10
        hit_mask = ~miss_mask
        return inters, normals, depth, hit_mask

    def trace_sdf_in_batch(self, rays_o, rays_d, batch_size=512**2, cpu=False):
        inters, normals, depth, hit_mask = [], [], [], []
        rn = rays_o.shape[0]
        for ri in range(0, rn, batch_size):
            inters_cur, normals_cur, depth_cur, hit_mask_cur = self.trace_sdf_with_mesh(rays_o[ri:ri+batch_size], rays_d[ri:ri+batch_size], 32, 9)
            if cpu:
                inters_cur = inters_cur.cpu()
                normals_cur = normals_cur.cpu()
                depth_cur = depth_cur.cpu()
                hit_mask_cur = hit_mask_cur.cpu()
            inters.append(inters_cur)
            normals.append(normals_cur)
            depth.append(depth_cur)
            hit_mask.append(hit_mask_cur)
        return torch.cat(inters, 0), torch.cat(normals, 0), torch.cat(depth, 0), torch.cat(hit_mask, 0)

    def get_intersection_around_mesh(self, sdf_fun, inv_fun, rays_o, rays_d, m_depth, sn0=128, sn1=9, perturb=False):
        """
        :param sdf_fun:
        :param inv_fun:
        :param rays_o:   pn,3
        :param rays_d:   pn,3
        :param m_depth:  pn,1
        :param sn0:
        :param sn1:
        :return: pn, sn1-1
        """
        pn, _ = rays_o.shape
        near, far = self.near_far_from_sphere(rays_o, rays_d)           # pn, 1
        unit_size = self.unit_size.to(rays_o.device) # 1,  
        
        t_min = (m_depth - unit_size * 4).clamp(min=near, max=far)  # pn, 1
        t_max = (m_depth + unit_size * 4).clamp(min=near, max=far)

        # Up sample
        with torch.no_grad():
            z_vals = torch.linspace(0.0, 1.0, sn0).to(rays_o.device)  # sn
            z_vals = t_min + (t_max - t_min) * z_vals[None, :]  # pn,sn
            if perturb > 0:
                t_rand = (torch.rand([pn, 1]) - 0.5)
                z_vals = z_vals + t_rand * 2.0 / sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, rays_o, rays_d) # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True) # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, rays_o, rays_d) # pn,sn1-1
            z_vals_mid = (z_vals_new[:,1:] + z_vals_new[:,:-1]) * 0.5
            
        hit_z_vals = z_vals_mid
        hit_weights = weights
        hit_sdf = mid_sdf
        return hit_z_vals, hit_weights, hit_sdf
                
    def trace_sdf_with_mesh(self, rays_o, rays_d, sn0, sn1):
        """
        Args:
            rays_o (_type_): pn, 3
            rays_d (_type_): pn, 3
        Returns:
            pn, -1
            TODO: whether to avg sn1 normals to surface normal
        """
        inters, normals, depth, hit_mask = self.trace(rays_o, rays_d)
        hit_mask = hit_mask.squeeze(-1)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if torch.sum(hit_mask) > 0:
            hit_z_vals, hit_weights, hit_sdf = \
                self.get_intersection_around_mesh(self.sdf_inter_fun, self.deviation_net, rays_o[hit_mask], rays_d[hit_mask], depth[hit_mask], sn0, sn1) # pn, sn1-1
            hit_weights = hit_weights / torch.sum(hit_weights, dim=-1, keepdim=True)
            hit_weights = torch.where(torch.isnan(hit_weights), torch.full_like(hit_weights, 1./(sn1-1)), hit_weights)
            depth[hit_mask] = torch.sum(hit_weights * hit_z_vals, -1, keepdim=True)                                 # pn, 1
            inters[hit_mask] = rays_o[hit_mask] + depth[hit_mask] * rays_d[hit_mask]                                # pn, 3
            gradient, _ = self.sdf_network.gradient(inters[hit_mask], None)
            normals[hit_mask] = F.normalize(gradient, dim=-1)
        # torch.set_default_tensor_type('torch.FloatTensor')
        return inters, normals, depth, hit_mask.unsqueeze(-1)
   
    def near_far_from_sphere(self, rays_o, rays_d):
        radius = self.radius if self.radius is not None else 1.0
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        mid = mid.to(rays_o.device)
        radius = radius.to(rays_o.device)
        near = mid - radius
        far = mid + radius
        near = torch.clamp(near, min=1e-3)
        return near, far

    def _warn_ray_tracing(self, centers):
        centers = centers.reshape([-1,3])
        distance = torch.norm(centers,dim=-1) + 1.0
        max_dist = torch.max(distance).cpu().numpy()
        if max_dist>10.0:
            print(f'warning!!! the max distance from the camera is {max_dist:.4f}, which is beyond 10.0 for the ray tracer')

    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0,2,1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3], device=poses.device).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3])  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    def _construct_ray_batch(self, imgs_info, device='cpu'):
        imn, _, h, w = imgs_info['imgs'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w)),-1)[:,:,(1,0)] # h,w,2
        coords = coords.to('cpu')
        coords = coords.float()[None,:,:,:].repeat(imn,1,1,1) # imn,h,w,2
        coords = coords.reshape(imn,h*w,2)
        coords = torch.cat([coords+0.5, torch.ones(imn, h*w, 1, dtype=torch.float32, device='cpu')], 2) # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        rays_d = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1)
        poses = imgs_info['poses']  # imn,3,4
        R, t = poses[:, :, :3], poses[:, :, 3:]
        rays_d = rays_d @ R
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = -R.permute(0,2,1) @ t # imn,3,3 @ imn,3,1
        self._warn_ray_tracing(rays_o)
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn,h*w,3

        human_poses = self.get_human_coordinate_poses(poses) # imn,3,4
        human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4
        rgb = imgs_info['imgs'].reshape(imn,3,h*w).permute(0,2,1) # imn,h*w,3

        assert imn==1
        ray_batch={
            'rays_o': rays_o[0].to(device),
            'rays_d': rays_d[0].to(device),
            'human_poses': human_poses[0].to(device),
            'rgb': rgb[0].to(device),
        }
        return ray_batch

    def _construct_ray_batch_nerf(self, imgs_info, device='cpu'):
        imn, _, h, w = imgs_info['imgs'].shape
        i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) # h, w, 3
        rays_d = dirs[None, ...].repeat(imn, 1, 1, 1).reshape(imn, h*w, 3).to(device) # imn, h*w, 3
        poses = imgs_info['poses'][:, :3, :].to(device)  # imn, 3, 4
        
        R, t = poses[:, :, :3], poses[:, :, 3:]
        rays_d = (R @ rays_d.permute(0, 2, 1)).permute(0, 2, 1)    # imn, 3, 3 @ imn, 3, h*w -> imn, 3, h*w -> imn, h*w, 3
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = t # imn, 3, 1
        self._warn_ray_tracing(rays_o)
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn, h*w, 3
        
        human_poses = self.get_human_coordinate_poses(poses) # imn,3,4
        human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4
        rgb = imgs_info['imgs'].reshape(imn,3,h*w).permute(0,2,1) # imn,h*w,3

        assert imn==1
        ray_batch={
            'rays_o': rays_o[0].to(device),
            'rays_d': rays_d[0].to(device),
            'human_poses': human_poses[0].to(device),
            'rgb': rgb[0].to(device),
        }
        return ray_batch


    def _get_trace_ray_batch_info(self, ray_batch_infos, is_train=True):
        rays_o, rays_d = ray_batch_infos['rays_o'], ray_batch_infos['rays_d']
        pn, device = rays_o.shape[0], rays_o.device
        
        inters, normals, depth, hit_mask = self.trace_sdf_in_batch(rays_o, rays_d) # imn
        inters, normals, depth, hit_mask = inters.reshape(pn,3), normals.reshape(pn,3), depth.reshape(pn,1), hit_mask.reshape(pn)

        if is_train:
            for k, v in ray_batch_infos.items():
                ray_batch_infos[k] = v[hit_mask]
            inter_ray_batch={
                'inters': inters[hit_mask].to(device),
                'normals': normals[hit_mask].to(device),
                'depth': depth[hit_mask].to(device),
            }
        else:
            inter_ray_batch={
                'inters': inters.to(device),
                'normals': normals.to(device),
                'depth': depth.to(device),
                'hit_mask': hit_mask.to(device),
            }
        ray_batch_infos.update(inter_ray_batch)
        return ray_batch_infos
        
        
    def _shuffle_train_ids(self):
        self.train_ids_i = 0 
        shuffle_idxs = torch.randperm(len(self.train_ids), device='cpu') # shuffle
        self.train_ids = self.train_ids[shuffle_idxs] 
        
    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.tbn, device='cpu') # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def shade(self, pts, view_dirs, normals, human_poses, is_train, step=None):
        rgb_pr, outputs = self.shader_network(pts, view_dirs, normals, human_poses, step, is_train)
        outputs['rgb_pr'] = rgb_pr
        return outputs

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def compute_diffuse_light_regularization(self, diffuse_lights):
        diffuse_white_reg = torch.sum(torch.abs(diffuse_lights - torch.mean(diffuse_lights, dim=-1, keepdim=True)), dim=-1)
        return diffuse_white_reg * self.cfg['reg_diffuse_light_lambda']

    def train_step(self, step):
        info_index = self.train_ids[self.train_ids_i]
        train_imgs_info = imgs_info_slice(self.train_imgs_info, torch.from_numpy(np.asarray([info_index],np.int64)))
        if self.cfg['nerfDataType']:
            ray_batch_info = self._construct_ray_batch_nerf(train_imgs_info, device='cuda')
        else:
            ray_batch_info = self._construct_ray_batch(train_imgs_info, device='cuda')

        # shuffle ray_batch
        _, _, h, w = train_imgs_info['imgs'].shape
        tbn = h*w
        rn = self.cfg['train_ray_num']
        shuffle_idxs = torch.randperm(tbn, device='cpu')
        choose_idxs = shuffle_idxs[:rn]
        for k, v in ray_batch_info.items():
            ray_batch_info[k] = v[choose_idxs]
        
        ray_batch_info = self._get_trace_ray_batch_info(ray_batch_info)

        pts = ray_batch_info['inters']
        view_dirs = -ray_batch_info['rays_d']
        normals = ray_batch_info['normals']
        rgb_gt = ray_batch_info['rgb']
        human_poses = ray_batch_info['human_poses']

        shade_outputs = self.shade(pts, view_dirs, normals, human_poses, True, step)
        shade_outputs['rgb_gt'] = rgb_gt
        shade_outputs['loss_rgb'] = self.compute_rgb_loss(shade_outputs['rgb_pr'],shade_outputs['rgb_gt'])
        if self.cfg['reg_mat']:
            shade_outputs['loss_mat_reg'] = self.shader_network.material_regularization(
                pts, normals, shade_outputs['metallic'], shade_outputs['roughness'], shade_outputs['albedo'], step)
        if self.cfg['reg_diffuse_light']:
            shade_outputs['loss_diffuse_light'] = self.compute_diffuse_light_regularization(shade_outputs['diffuse_light'])

        self.train_ids_i += 1        
        if self.train_ids_i >= self.train_num : self._shuffle_train_ids()
        return shade_outputs

    def test_step(self, index):
        test_imgs_info = imgs_info_slice(self.test_imgs_info, torch.from_numpy(np.asarray([index],np.int64)))
        _, _, h, w = test_imgs_info['imgs'].shape
        if self.cfg['nerfDataType']:
            ray_batch = self._construct_ray_batch_nerf(test_imgs_info, 'cuda')
        else:
            ray_batch = self._construct_ray_batch(test_imgs_info, 'cuda')
        trn = self.cfg['test_ray_num']

        output_keys = {'rgb_gt':3, 'rgb_pr':3, 
                       'specular_light':3, 'specular_color':3, 
                       'diffuse_light':3, 'diffuse_color':3, 
                       'albedo':3, 'metallic':1, 'roughness':1,
                       'occ_trace':1, 'indirect_light':3}
        outputs = {k:[] for k in output_keys.keys()}
        rn = ray_batch['rays_o'].shape[0]
        for ri in range(0, rn, trn):
            test_ray_batch = {}
            for k, v in ray_batch.items():
                test_ray_batch[k] = v[ri:ri+trn]
            test_ray_batch = self._get_trace_ray_batch_info(test_ray_batch, is_train=False)
            
            hit_mask = test_ray_batch['hit_mask']
            outputs_cur = {k: torch.zeros(hit_mask.shape[0], d) for k, d in output_keys.items()}
            if torch.sum(hit_mask)>0:
                pts = test_ray_batch['inters'][hit_mask]
                view_dirs = -test_ray_batch['rays_d'][hit_mask]
                normals = test_ray_batch['normals'][hit_mask]
                rgb_gt = test_ray_batch['rgb'][hit_mask]
                human_poses = test_ray_batch['human_poses'][hit_mask]

                shade_outputs = self.shade(pts, view_dirs, normals, human_poses, False)

                outputs_cur['rgb_pr'][hit_mask] = shade_outputs['rgb_pr']
                outputs_cur['rgb_gt'][hit_mask] = rgb_gt
                outputs_cur['specular_light'][hit_mask] = shade_outputs['specular_light']
                outputs_cur['specular_color'][hit_mask] = shade_outputs['specular_color']
                outputs_cur['diffuse_color'][hit_mask] = shade_outputs['diffuse_color']
                outputs_cur['diffuse_light'][hit_mask] = shade_outputs['diffuse_light']
                outputs_cur['albedo'][hit_mask] = shade_outputs['albedo']
                outputs_cur['metallic'][hit_mask] = shade_outputs['metallic']
                outputs_cur['roughness'][hit_mask] = torch.sqrt(shade_outputs['roughness']) # note: we assume predictions are roughness squared
                outputs_cur['occ_trace'][hit_mask] = shade_outputs['visibility']
                outputs_cur['indirect_light'][hit_mask] = shade_outputs['indirect_light']

            for k in output_keys.keys():
                outputs[k].append(outputs_cur[k])

        for k in output_keys.keys():
            outputs[k] = torch.cat(outputs[k], 0).reshape(h, w, -1)

        return outputs

    def nvs(self, pose, K, h, w):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        poses = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)
        
        def construct_ray_dirs_nerf():
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
            i = i.t().to(device)
            j = j.t().to(device)
            k = K[0]
            dirs = torch.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -torch.ones_like(i).to(device)], -1) # h, w, 3
            rays_d = dirs[None, ...].repeat(1, 1, 1, 1).reshape(1, h*w, 3) # imn, h*w, 3
            pose = poses[:, :3, :]      # imn, 3, 4
            
            R, t = pose[:, :, :3], pose[:, :, 3:]
            rays_d = (R @ rays_d.permute(0, 2, 1)).permute(0, 2, 1)    # imn, 3, 3 @ imn, 3, h*w -> imn, 3, h*w -> imn, h*w, 3
            rays_d = F.normalize(rays_d, dim=-1)
            rays_o = t # imn, 3, 1
            self._warn_ray_tracing(rays_o)
            rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn, h*w, 3

            human_poses = self.get_human_coordinate_poses(pose) # imn,3,4
            human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4

            ray_batch={
                'rays_o': rays_o[0].to(device),
                'rays_d': rays_d[0].to(device),
                'human_poses': human_poses[0].to(device),
            }
            
            return ray_batch
            
        def construct_ray_dirs():
            coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
            coords = coords.to(device)
            coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
            coords = coords.reshape(1, h * w, 2)
            coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3
            # 1,h*w,3 @ imn,3,3 => 1,h*w,3
            dirs = coords @ torch.inverse(K).permute(0, 2, 1)
            rays_d = F.normalize(dirs, dim=-1).reshape(-1, 3)     # rn,3
            rays_d = poses[:, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
            rays_d = rays_d[..., 0]  # rn,3
            rays_d = F.normalize(rays_d, dim=-1)
            
            rays_o = (poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:])[:, :, 0]        # 1, 3
            rays_o = rays_o.repeat(h*w, 1)   # rn, 3

            human_poses = self.get_human_coordinate_poses(poses) # imn,3,4
            human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4
                                   
            ray_batch = {
                'rays_o': rays_o.to(device),
                'rays_d': rays_d.to(device),       
                'human_poses': human_poses[0].to(device),         
            }
            return ray_batch

        if self.cfg['nerfDataType']:
            ray_batch = construct_ray_dirs_nerf()
        else:
            ray_batch = construct_ray_dirs()
                 
        rn = h * w
        trn = 512
        output_keys = {'color':3, 'normal':3, 'spec_light':3, 'diff_light':3, 'indirect_light':3, 'spec_color':3, 'diff_color':3, 'albedo':3, 'roughness':1, 'metallic':1, 'occ_trace':1,}
        outputs = {k:[] for k in output_keys.keys()}
        for ri in range(0, rn, trn):
            nvs_ray_batch = {}
            for k, v in ray_batch.items():
                nvs_ray_batch[k] = v[ri:ri+trn]
            nvs_ray_batch = self._get_trace_ray_batch_info(nvs_ray_batch, is_train=False)
            
            hit_mask = nvs_ray_batch['hit_mask']
            outputs_cur = {k: torch.zeros(hit_mask.shape[0], d) for k, d in output_keys.items()}
            if torch.sum(hit_mask)>0:
                pts = nvs_ray_batch['inters'][hit_mask]
                view_dirs = -nvs_ray_batch['rays_d'][hit_mask]
                normals = nvs_ray_batch['normals'][hit_mask]
                human_poses = nvs_ray_batch['human_poses'][hit_mask]

                shade_outputs = self.shade(pts, view_dirs, normals, human_poses, False)

                outputs_cur['color'][hit_mask] = shade_outputs['rgb_pr']
                outputs_cur['normal'][hit_mask] = normals
                outputs_cur['normal'][~hit_mask] = torch.tensor([0., 0., 1.], device=hit_mask.device)
                outputs_cur['spec_light'][hit_mask] = shade_outputs['specular_light']
                outputs_cur['diff_light'][hit_mask] = shade_outputs['diffuse_light']
                outputs_cur['indirect_light'][hit_mask] = shade_outputs['indirect_light']
                outputs_cur['occ_trace'][hit_mask] = shade_outputs['visibility']
                
                outputs_cur['spec_color'][hit_mask] = shade_outputs['specular_color']
                outputs_cur['diff_color'][hit_mask] = shade_outputs['diffuse_color']

                outputs_cur['albedo'][hit_mask] = shade_outputs['albedo']
                outputs_cur['roughness'][hit_mask] = torch.sqrt(shade_outputs['roughness']) # note: we assume predictions are roughness squared
                outputs_cur['metallic'][hit_mask] = shade_outputs['metallic']
                
            outputs_cur['color'][~hit_mask] = torch.tensor([1., 1., 1.], device=hit_mask.device) 
            for k in output_keys.keys():
                outputs[k].append(outputs_cur[k])
        
        for k in output_keys.keys():
            outputs[k] = torch.cat(outputs[k], 0).reshape(h, w, -1).detach().cpu().numpy()

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs
    
    def forward(self, data):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        is_train = 'eval' not in data
        step = data['step']

        if is_train:
            outputs = self.train_step(step)
        else:
            index = data['index']
            outputs = self.test_step(index)

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def predict_materials(self, batch_size=8192):
        verts = torch.from_numpy(np.asarray(self.mesh.vertices,np.float32)).cuda().float()
        metallic, roughness, albedo = [], [], []
        for vi in range(0, verts.shape[0], batch_size):
            m, r, a = self.shader_network.predict_materials(verts[vi:vi+batch_size])
            r = torch.sqrt(torch.clamp(r, min=1e-7)) # note: we assume predictions are squared roughness!!!
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {'metallic': np.concatenate(metallic, 0),
                'roughness': np.concatenate(roughness, 0),
                'albedo': np.concatenate(albedo, 0)}
        
    def check_sdf_trace(self, pose, K, h, w):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        poses = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)

        def construct_ray_dirs():
            coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
            coords = coords.to(device)
            coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
            coords = coords.reshape(1, h * w, 2)
            coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3
            # 1,h*w,3 @ 1,3,3 => 1,h*w,3
            rays_d = coords @ torch.inverse(K).permute(0, 2, 1)
            R, t = poses[:, :, :3], poses[:, :, 3:]            
            rays_d = rays_d @ R
            rays_d = F.normalize(rays_d, dim=-1)
            rays_o = -R.permute(0,2,1) @ t # 1,3,3 @ 1,3,1
            self._warn_ray_tracing(rays_o)
            rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # 1,h*w,3

            inters, normals, depth, hit_mask = self.trace_sdf_in_batch(rays_o.reshape(-1,3), rays_d.reshape(-1,3), cpu=False) # imn
            inters, normals, depth, hit_mask = inters.reshape(1,h*w,3), normals.reshape(1,h*w,3), depth.reshape(1,h*w,1), hit_mask.reshape(1, h*w)
            
            human_poses = self.get_human_coordinate_poses(poses) # imn,3,4
            human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4
                                   
            ray_batch={
                'rays_o': rays_o[0].to(device),
                'rays_d': rays_d[0].to(device),
                'inters': inters[0].to(device),
                'normals': normals[0].to(device),
                'depth': depth[0].to(device),
                'human_poses': human_poses[0].to(device),
                'hit_mask': hit_mask[0].to(device),
            }
            return ray_batch
        
        def construct_ray_dirs_nerf():
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
            i = i.t().to(device)
            j = j.t().to(device)
            k = K[0]
            dirs = torch.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -torch.ones_like(i).to(device)], -1) # h, w, 3
            rays_d = dirs[None, ...].repeat(1, 1, 1, 1).reshape(1, h*w, 3) # imn, h*w, 3
            pose = poses[:, :3, :]      # imn, 3, 4
            
            R, t = pose[:, :, :3], pose[:, :, 3:]
            rays_d = (R @ rays_d.permute(0, 2, 1)).permute(0, 2, 1)    # imn, 3, 3 @ imn, 3, h*w -> imn, 3, h*w -> imn, h*w, 3
            rays_d = F.normalize(rays_d, dim=-1)
            rays_o = t # imn, 3, 1
            self._warn_ray_tracing(rays_o)
            rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn, h*w, 3
            
            inters, normals, depth, hit_mask = self.trace_sdf_in_batch(rays_o.reshape(-1,3), rays_d.reshape(-1,3), batch_size=128**2, cpu=False) # imn
            inters, normals, depth, hit_mask = inters.reshape(1,h*w,3), normals.reshape(1,h*w,3), depth.reshape(1,h*w,1), hit_mask.reshape(1, h*w)

            human_poses = self.get_human_coordinate_poses(pose) # imn,3,4
            human_poses = human_poses.unsqueeze(1).repeat(1,h*w,1,1) # imn,h*w,3,4

            ray_batch={
                'rays_o': rays_o[0].to(device),
                'rays_d': rays_d[0].to(device),
                'inters': inters[0].to(device),
                'normals': normals[0].to(device),
                'depth': depth[0].to(device),
                'human_poses': human_poses[0].to(device),
                'hit_mask': hit_mask[0].to(device),
            }
            
            return ray_batch
            
        if self.cfg['nerfDataType']:
            ray_batch = construct_ray_dirs_nerf()
        else:
            ray_batch = construct_ray_dirs()
                 
        rn = h * w
        trn = 2048
        output_keys = {'normal' : 3, 'mask' : 1}
        outputs = {k:[] for k in output_keys.keys()}
        for ri in range(0, rn, trn):
            hit_mask = ray_batch['hit_mask'][ri:ri+trn]
            outputs_cur = {k: torch.zeros(hit_mask.shape[0], d) for k, d in output_keys.items()}
            if torch.sum(hit_mask)>0:
                pts = ray_batch['inters'][ri:ri+trn][hit_mask]
                view_dirs = -ray_batch['rays_d'][ri:ri+trn][hit_mask]
                normals = ray_batch['normals'][ri:ri+trn][hit_mask]
                
                outputs_cur['normal'][hit_mask] = normals
                outputs_cur['mask'][hit_mask] = 1.

            outputs_cur['normal'][~hit_mask] = torch.tensor([0., 0., 1.], device=hit_mask.device)
            for k in output_keys.keys():
                outputs[k].append(outputs_cur[k])
                
            cur_ray_batch = {}
            for k, v in ray_batch.items(): cur_ray_batch[k] = v[ri:ri + trn]
        
        for k in output_keys.keys():
            outputs[k] = torch.cat(outputs[k], 0).reshape(h, w, -1).detach().cpu().numpy()

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs