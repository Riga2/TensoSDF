import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.raw_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn
from utils.network_utils import get_embedder, get_sphere_intersection, saturate_dot, IPE, get_camera_plane_intersection, contraction
from network.other_field import GaussianBlur1D, GaussianBlur2D
import nvdiffrast.torch as dr
from network.other_field import make_predictor_3layer, make_predictor_4layer
from utils.base_utils import az_el_to_points, sample_sphere

class TensoSDF(nn.Module):
    def __init__(self, gridSize, aabb, device='cuda', sdf_n_comp=36, sdf_dim = 256, app_dim = 128, init_n_levels = 3):
        super().__init__()
        self.sdf_n_comp = sdf_n_comp
        self.sdf_dim = sdf_dim
        self.app_dim = app_dim
        self.device = device

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.nplane = len(self.vecMode)
        self.init_radius = 0.2

        self.kernel_size = 5
        self.sigma = 0.5
        self.define_gaussian(self.kernel_size, self.sigma)

        self.update_gridSize_aabb(gridSize, aabb, init_n_levels)

        self.init_svd_volume(device)
        self.init_mlp(device, sdf_multires=3, sdf_feat_multires=0)


    def define_gaussian(self, kernel_size=5, sigma=0.8, stride=1):
        print(f"Gaussian settings: {kernel_size}, {sigma}")
        self.gaussian1d = GaussianBlur1D(kernel_size=kernel_size, sigma=sigma, stride=stride)
        self.gaussian2d = GaussianBlur2D(kernel_size=kernel_size, sigma=sigma, stride=stride)

    def gaussian_conv(self):
        self.sdf_plane_gaussian, self.sdf_line_gaussian = [], []
        for i in range(self.nplane):
            self.sdf_plane_gaussian.append(self.gaussian2d(self.sdf_plane[i].permute(1,0,2,3)).permute(1,0,2,3))
            self.sdf_line_gaussian.append(self.gaussian1d(self.sdf_line[i].permute(1,0,2,3).squeeze(-1)).unsqueeze(-1).permute(1,0,2,3))

    def update_gridSize_aabb(self, gridSize, aabb, n_levels):
        self.gridSize = gridSize
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.units = self.aabbSize / (self.gridSize - 1)
        self.n_levels = n_levels
        print(f"current levels : {self.n_levels}, current units : {self.units}")

    def init_svd_volume(self, device):
        self.sdf_plane, self.sdf_line = self.circle_init_one_svd(self.sdf_n_comp, device)

    def init_mlp(self, device, sdf_multires, sdf_feat_multires=0):
        self.embed_fn = None
        sdf_input_ch = 3
        if sdf_multires > 0:
            self.embed_fn, sdf_input_ch = get_embedder(sdf_multires, input_dims=sdf_input_ch)

        self.sdf_feat_embed_fn = None
        sdf_feat_input_ch = self.sdf_n_comp * self.nplane
        if sdf_feat_multires > 0:
            self.sdf_feat_embed_fn, sdf_feat_input_ch = get_embedder(sdf_feat_multires, input_dims=sdf_feat_input_ch)
        
        self.sdf_mat = nn.Sequential(
            nn.Linear(sdf_feat_input_ch + sdf_input_ch, self.sdf_dim), nn.Softplus(beta=100),
            nn.Linear(self.sdf_dim , 1 + self.app_dim)
        ).to(device)

        torch.nn.init.constant_(self.sdf_mat[0].bias, 0.0)
        torch.nn.init.normal_(self.sdf_mat[0].weight, 0.0, np.sqrt(2) / np.sqrt(self.sdf_dim))
        torch.nn.init.constant_(self.sdf_mat[-1].bias, -self.init_radius)
        torch.nn.init.normal_(self.sdf_mat[-1].weight, mean=np.sqrt(np.pi) / np.sqrt(self.sdf_dim), std=0.0001)

    def init_mat_mlp(self, mat_multires=6):
        out_dim = 3 + 1 + 1
        self.mat_pos_embed_fn = None
        mat_pos_input_ch = 3
        if mat_multires > 0:
            self.mat_pos_embed_fn, mat_pos_input_ch = get_embedder(mat_multires, input_dims=mat_pos_input_ch)
        self.material_mlp = make_predictor_3layer(self.app_dim + mat_pos_input_ch, out_dim, run_dim=128)
         
    def circle_init_one_svd(self, n_component, device):
        plane_coef, line_coef = [], []
        for i in range(self.nplane):
            planeSize = self.gridSize[self.matMode[i]]
            lineSize = self.gridSize[self.vecMode[i]]
            init_plane = self.circle_init(planeSize).expand(n_component, planeSize[0], planeSize[1]).unsqueeze(0) # 1, n, grid, grid
            init_line = torch.ones((1, n_component, lineSize, 1)) * (1./(n_component * self.nplane)) # 1, n, grid, 1
            plane_coef.append(nn.Parameter(init_plane.clone()))
            line_coef.append(nn.Parameter(init_line.clone()))
        
        return nn.ParameterList(plane_coef).to(device), nn.ParameterList(line_coef).to(device)

    def random_init_one_svd(self, n_component, device):
        plane_coef, line_coef = [], []
        for i in range(self.nplane):
            planeSize = self.gridSize[self.matMode[i]]
            lineSize = self.gridSize[self.vecMode[i]]
            init_plane = 0.1 * torch.randn((1, n_component, planeSize[0], planeSize[1])) # 1, n, grid, grid
            init_line =  0.1 * torch.randn((1, n_component, lineSize, 1)) # 1, n, grid, 1
            plane_coef.append(nn.Parameter(init_plane.clone()))
            line_coef.append(nn.Parameter(init_line.clone()))
        
        return nn.ParameterList(plane_coef).to(device), nn.ParameterList(line_coef).to(device)

    def circle_init(self, gridSize):
        x = torch.linspace(-1, 1, gridSize[0])
        y = torch.linspace(-1, 1, gridSize[1])
        x, y = torch.meshgrid(x, y)
        pts = torch.stack([x, y], dim=-1)
        init_sdf = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True) - self.init_radius
        return init_sdf.permute(2, 0, 1)  # (1, grid_sz, grid_sz)   
        
    def TV_loss_sdf(self, reg):
        total = 0
        for i in range(self.nplane):
            total += reg(self.sdf_plane[i])
            total += reg(self.sdf_line[i])
        return total

    def TV_loss_app(self, reg):
        return torch.zeros(1)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.sdf_line, 'lr': lr_init_spatialxyz}, {'params': self.sdf_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.sdf_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def sdf(self, xyz_sampled, level_vol=None):
        return self.forward(xyz_sampled, level_vol)[..., :1]

    def sdf_hidden_appearance(self, xyz_sampled, level_vol):
        return self.forward(xyz_sampled, level_vol)[..., 1:]

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
                # F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='nearest'))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
                # F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='nearest'))
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.define_gaussian(self.kernel_size[self.curIndex], self.sigma[self.curIndex])
        # self.curIndex += 1
        new_levels = self.n_levels + 1
        res_target = ((res_target / 2**(new_levels - 1)).int() * 2**(new_levels - 1)) # can be divided by new_levels - 1
        self.sdf_plane, self.sdf_line = self.up_sampling_VM(self.sdf_plane, self.sdf_line, res_target)

        self.update_gridSize_aabb(res_target, self.aabb, new_levels)
        print(f'upsamping to {res_target}, remember to update renderer')
        return res_target, self.n_levels

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(self.nplane):
            mode0 = self.vecMode[i]
            self.sdf_line[i] = torch.nn.Parameter(
                self.sdf_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.sdf_plane[i] = torch.nn.Parameter(
                self.sdf_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )

        newSize = b_r - t_l
        self.update_gridSize_aabb(newSize, new_aabb)

        print(f'Shrink to {new_aabb}, remember to update renderer')
        return (newSize[0], newSize[1], newSize[2])
    
    @torch.no_grad()
    def compute_sample_level(self, x, k = 0.1):
        sdf = self.sdf(x)
        eps = self.units
        eps_x = torch.tensor([eps[0], 0., 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0., eps[1], 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0., 0., eps[2]], dtype=x.dtype, device=x.device)  # [3]
        sdf_x_pos = self.sdf(x + eps_x)  # [...,1]
        sdf_x_neg = self.sdf(x - eps_x)  # [...,1]
        sdf_y_pos = self.sdf(x + eps_y)  # [...,1]
        sdf_y_neg = self.sdf(x - eps_y)  # [...,1]
        sdf_z_pos = self.sdf(x + eps_z)  # [...,1]
        sdf_z_neg = self.sdf(x - eps_z)  # [...,1]
        delta_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf)
        delta_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf)
        delta_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf)
        delta_mean = torch.sum(torch.cat([delta_xx, delta_yy, delta_zz], dim=-1).abs(), dim=-1, keepdim=True) / (3 * eps.mean()) # [...,1]
        # level = torch.clamp(-torch.log2((delta_mean + 0.00001) / k), min=0., max=self.n_levels-1) # [...,1]
        # print(f"level min:{level.min()}, max:{level.max()}")
        # print(delta_mean.max())
        return torch.clamp(1. - delta_mean, min=0., max=1.)
        
    def gradient(self, x, level_vol, training=False, sdf=None):
        eps = self.units
        # 1st-order gradient
        eps_x = torch.tensor([eps[0], 0., 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0., eps[1], 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0., 0., eps[2]], dtype=x.dtype, device=x.device)  # [3]
        sdf_x_pos = self.sdf(x + eps_x, level_vol)  # [...,1]
        sdf_x_neg = self.sdf(x - eps_x, level_vol)  # [...,1]
        sdf_y_pos = self.sdf(x + eps_y, level_vol)  # [...,1]
        sdf_y_neg = self.sdf(x - eps_y, level_vol)  # [...,1]
        sdf_z_pos = self.sdf(x + eps_z, level_vol)  # [...,1]
        sdf_z_neg = self.sdf(x - eps_z, level_vol)  # [...,1]
        gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps[0])
        gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps[1])
        gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps[2])
        gradients = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
        # 2nd-order gradient (hessian)
        if training:
            assert sdf is not None  # computed when feed-forwarding through the network
            hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps[0] ** 2)  # [...,1]
            hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps[1] ** 2)  # [...,1]
            hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps[2] ** 2)  # [...,1]
            hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
            normal_hessian = (gradients * hessian).sum(dim=-1) / (torch.sum(gradients ** 2, dim=-1) + 1e-5)
        else:
            normal_hessian = None
            # hessian = None
        return gradients, normal_hessian

    def forward(self, xyz_sampled, level_vol):
        # xyz_sampled : (rn*sn, 3)
        # plane + line basis
        xyz_sampled = contraction(xyz_sampled, self.aabb).reshape(-1, 3)
        level = (torch.zeros([xyz_sampled.shape[0], 1], device=xyz_sampled.device) if level_vol is None else level_vol).view(-1, 1).unsqueeze(0).contiguous() # 1, N, 1
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)          # 3, rn * sn, 1, 2

        plane_coef_point,line_coef_point = [],[]
        planes, lines = self.sdf_plane, self.sdf_line
        for idx in range(self.nplane):
            plane_coef_point.append(
                dr.texture(planes[idx].permute(0, 2, 3, 1).contiguous(), 
                           coordinate_plane[[idx]], 
                           mip_level_bias=level, 
                           boundary_mode="clamp", 
                           max_mip_level=self.n_levels-1
                           ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(
                dr.texture(lines[idx].permute(0, 2, 3, 1).contiguous(), 
                           coordinate_line[[idx]], 
                           mip_level_bias=level, 
                           boundary_mode="clamp", 
                           max_mip_level=self.n_levels-1
                           ).permute(0, 3, 1, 2).contiguous().view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        sigma_feature = plane_coef_point * line_coef_point

        inputs_xyz = xyz_sampled
        inputs_feat = sigma_feature.T
        if self.embed_fn is not None:
            inputs_xyz = self.embed_fn(xyz_sampled)
        if self.sdf_feat_embed_fn is not None:
            inputs_feat = self.sdf_feat_embed_fn(sigma_feature.T)
        out_feats = self.sdf_mat(torch.cat([inputs_feat, inputs_xyz], dim=-1))       # (rn * sn, 1)
        return out_feats

    def grid_gaussian_loss(self):
        total_loss = 0.
        k = self.kernel_size // 2
        for i in range(self.nplane):
            plane_gaussian = self.gaussian2d(self.sdf_plane[i].permute(1,0,2,3)).permute(1,0,2,3)
            line_gaussian = self.gaussian1d(self.sdf_line[i].permute(1,0,2,3).squeeze(-1)).unsqueeze(-1).permute(1,0,2,3)
            total_loss += torch.sum((self.sdf_plane[i][..., k:-k, k:-k] - plane_gaussian[..., k:-k, k:-k]).square())
            total_loss += torch.sum((self.sdf_line[i][..., k:-k, :] - line_gaussian[..., k:-k, :]).square())
        return total_loss 
    
    def predict_materials(self, feats, xyz_sampled):
        inputs_xyz = xyz_sampled
        if self.mat_pos_embed_fn is not None:
            inputs_xyz = self.mat_pos_embed_fn(inputs_xyz)
        materials = self.material_mlp(torch.cat([feats, xyz_sampled], -1))
        albedo, roughness, metallic = materials[..., :3], materials[..., 3:4], materials[..., 4:]        
        return albedo, roughness, metallic
    

class ShapeShadingNetwork(nn.Module):
    default_cfg={
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
        'app_feats_dim': 128,
        'has_radiance_field': False,
        'radiance_field_step': 0,
    }
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}
        feats_dim = self.cfg['app_feats_dim']

        # radiance MLPs
        if self.cfg['has_radiance_field']:
            self.init_rad_mlp(feats_dim, pos_multires=0, dir_multires=4)

        # material MLPs
        self.init_mat_mlp(feats_dim, pos_multires=3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor_3layer(72*2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor_3layer(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias,np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor_3layer(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor_3layer(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor_3layer(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def init_mat_mlp(self, feats_dim, pos_multires):
        out_dim = 3 + 1 + 1
        self.mat_pos_embed_fn = None
        pos_input_ch = 3
        if pos_multires > 0:
            self.mat_pos_embed_fn, pos_input_ch = get_embedder(pos_multires, input_dims=pos_input_ch)
        self.mat_mlp = make_predictor_3layer(feats_dim + pos_input_ch, out_dim, run_dim=128)

    def init_rad_mlp(self, feats_dim, pos_multires, dir_multires):
        out_dim = 3
        self.rad_pos_embed_fn = None
        pos_input_ch = 3
        if pos_multires > 0:
            self.rad_pos_embed_fn, pos_input_ch = get_embedder(pos_multires, input_dims=pos_input_ch)
        self.rad_dir_embed_fn = None
        dir_input_ch = 3
        if dir_multires > 0:
            self.rad_dir_embed_fn, dir_input_ch = get_embedder(dir_multires, input_dims=dir_input_ch)
        self.rad_mlp = make_predictor_3layer(feats_dim + pos_input_ch + dir_input_ch + 3, out_dim, run_dim=128)        
        
    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        direct_light = self.outer_light(ref_roughness)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1)) # this is occlusion prob
        occ_prob = occ_prob*0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob,min=0,max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, direct_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness = torch.ones([normals.shape[0],1])
        ref = self.sph_enc(normals, roughness)
        light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
        normals = F.normalize(normals, dim=-1)
        normals[normals[:, :2].sum(dim=-1) == 0.] = torch.tensor([0.0, 1e-6, 1.0], device=normals.device) # no zeros
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        # material
        mat_pts = points
        if self.mat_pos_embed_fn is not None:
            mat_pts = self.mat_pos_embed_fn(mat_pts)
        mat = self.mat_mlp(torch.cat([feature_vectors, mat_pts], -1))
        albedo, roughness, metallic = mat[..., :3], mat[..., 3:4], mat[..., 4:]

        # radiance
        if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
            rad_pts, rad_dirs = points, view_dirs
            if self.rad_pos_embed_fn is not None:
                rad_pts = self.rad_pos_embed_fn(rad_pts)
            if self.rad_dir_embed_fn is not None:
                rad_dirs = self.rad_dir_embed_fn(rad_dirs)
            radiance = self.rad_mlp(torch.cat([feature_vectors, rad_pts, rad_dirs, normals], -1))
        
        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light, occ_prob, indirect_light, direct_light, human_light = self.predict_specular_lights(points, feature_vectors, reflective, roughness, human_poses, step)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness,min=0.0,max=1.0)],-1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn//bn, bn, -1).contiguous(), filter_mode='linear', boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:,0:1] + fg_lookup[:,1:2])
        specular_color = specular_ref * specular_light

        # integrated together
        color = diffuse_color + specular_color

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)
        color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
            'roughness': roughness,
        }

        if inter_results:
            intermediate_results={
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_direct_light': direct_light,
                'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'roughness': roughness,
                'albedo' : albedo,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
            }
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            if self.cfg['has_radiance_field'] and step > self.cfg['radiance_field_step']:
                return color, radiance, occ_info
            else:
                return color, None, occ_info

    def predict_materials(self, points, feature_vectors):
        mat_pts = points
        if self.mat_pos_embed_fn is not None:
            mat_pts = self.mat_pos_embed_fn(mat_pts)
        mat = self.mat_mlp(torch.cat([feature_vectors, mat_pts], -1))
        albedo, roughness, metallic = mat[..., :3], mat[..., 3:4], mat[..., 4:]
        return metallic, roughness, albedo


class MaterialFeatsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, input_dim = get_embedder(8, 3)
        run_dim= 256
        self.module0=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
        )
        self.module1=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        input = x
        x = self.module0(x)
        return self.module1(torch.cat([x, input], -1))


class MCShadingNetwork(nn.Module):
    default_cfg={
        'diffuse_sample_num': 512,
        'specular_sample_num': 256,
        'human_lights': False,
        'light_exp_max': 5.0,
        'inner_light_exp_max': 5.0,
        'outer_light_version': 'direction',
        'geometry_type': 'schlick',

        'reg_change': True,
        'change_eps': 0.05,
        'change_type': 'gaussian',
        'reg_lambda1': 0.005,
        'reg_min_max': True,

        'random_azimuth': True,
        'is_real': False,
    }
    def __init__(self, cfg, ray_trace_fun):
        self.cfg={**self.default_cfg, **cfg}
        super().__init__()

        # material part
        self.feats_network = MaterialFeatsNetwork()
        self.metallic_predictor = make_predictor_4layer(256+3, 1)
        self.roughness_predictor = make_predictor_4layer(256+3, 1)
        self.albedo_predictor = make_predictor_4layer(256+3, 3)

        # light part
        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(8, 3)
        if self.cfg['outer_light_version']=='direction':
            self.outer_light = make_predictor_4layer(72, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        elif self.cfg['outer_light_version']=='sphere_direction':
            self.outer_light = make_predictor_4layer(72*2, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        else:
            raise NotImplementedError
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
        if self.cfg['human_lights']:
            self.human_light = make_predictor_4layer(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light[-2].bias, np.log(0.02))
        self.inner_light = make_predictor_4layer(pos_dim + 72, 3, activation='exp', exp_max=self.cfg['inner_light_exp_max'])
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))

        # predefined diffuse sample directions
        az, el = sample_sphere(self.cfg['diffuse_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(self.diffuse_direction_samples.astype(np.float32)).cuda() # [dn0,2]

        az, el = sample_sphere(self.cfg['specular_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.specular_direction_samples = np.stack([az, el], -1)
        self.specular_direction_samples = torch.from_numpy(self.specular_direction_samples.astype(np.float32)).cuda() # [dn1,2]

        az, el = sample_sphere(8192, 0)
        light_pts = az_el_to_points(az, el)
        self.register_buffer('light_pts', torch.from_numpy(light_pts.astype(np.float32)))
        self.ray_trace_fun = ray_trace_fun

    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1) # pn,1
        otho0 = torch.cat([y,-x,torch.zeros_like(x)],-1)
        otho1 = torch.cat([-z,torch.zeros_like(x),x],-1)
        mask0 = torch.norm(otho0,dim=-1)>torch.norm(otho1,dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho

    def sample_diffuse_directions(self, normals, is_train):
        # normals [pn,3]
        z = normals # pn,3
        x = self.get_orthogonal_directions(normals) # pn,3
        y = torch.cross(z, x, dim=-1) # pn,3
        # y = torch.cross(z, x, dim=-1) # pn,3

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples,1,dim=1) # sn,1
        el, az = el.unsqueeze(0), az.unsqueeze(0)
        az = az * np.pi * 2
        el_sqrt = torch.sqrt(el+1e-7)
        if is_train and self.cfg['random_azimuth']:
            az = (az + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1) # pn,sn,3
        return directions

    def sample_specular_directions(self, reflections, roughness, is_train):
        # roughness [pn,1]
        z = reflections  # pn,3
        x = self.get_orthogonal_directions(reflections)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        a = roughness # we assume the predicted roughness is already squared

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = np.pi * 2 * az # sn,1
        a, el = a.unsqueeze(1), el.unsqueeze(0) # [pn,1,1] [1,sn,1]
        cos_theta = torch.sqrt((1.0 - el + 1e-6) / (1.0 + (a**2 - 1.0) * el + 1e-6) + 1e-6) # pn,sn,1
        sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-6) # pn,sn,1

        phi = phi.unsqueeze(0) # 1,sn,1
        if is_train and self.cfg['random_azimuth']:
            phi = (phi + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_x = torch.cos(phi) * sin_theta # pn,sn,1
        coeff_y = torch.sin(phi) * sin_theta # pn,sn,1
        coeff_z = cos_theta # pn,sn,1

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1) # pn,sn,3
        return directions

    def get_inner_lights(self, points, view_dirs, normals):
        pos_enc = self.pos_enc(points)
        normals = F.normalize(normals,dim=-1)
        view_dirs = F.normalize(view_dirs,dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        dir_enc = self.sph_enc(reflections, 0)
        return self.inner_light(torch.cat([pos_enc, dir_enc], -1))

    def predict_outer_lights(self, points, directions):
        if self.cfg['outer_light_version'] == 'direction':
            outer_enc = self.sph_enc(directions, 0)
            outer_lights = self.outer_light(outer_enc)
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            outer_dirs = directions
            outer_pts = points
            outer_enc = self.sph_enc(outer_dirs, 0)
            mask = torch.norm(outer_pts,dim=-1)>0.999
            if torch.sum(mask)>0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[mask]*=0.999 # shrink this point a little bit
            dists = get_sphere_intersection(outer_pts, outer_dirs)
            sphere_pts = outer_pts + outer_dirs * dists
            sphere_pts = self.sph_enc(sphere_pts, 0)
            outer_lights = self.outer_light(torch.cat([outer_enc, sphere_pts], -1))
        else:
            raise NotImplementedError
        return outer_lights

    def get_human_light(self, points, directions, human_poses):
        inter, dists, hits = get_camera_plane_intersection(points, directions, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits

        var = torch.zeros_like(mean)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def get_lights(self, points, directions, human_poses):
        # trace
        shape = points.shape[:-1] # pn,sn
        eps = 1e-5
        inters, normals, depth, hit_mask = self.ray_trace_fun(points.reshape(-1,3)+directions.reshape(-1,3) * eps, directions.reshape(-1,3))
        inters, normals, depth, hit_mask = inters.reshape(*shape,3), normals.reshape(*shape,3), depth.reshape(*shape, 1), hit_mask.reshape(*shape)
        miss_mask = ~hit_mask

        # hit_mask
        lights = torch.zeros(*shape, 3)
        human_lights, human_weights = torch.zeros([1,3]), torch.zeros([1,1])
        if torch.sum(miss_mask)>0:
            outer_lights = self.predict_outer_lights(points[miss_mask], directions[miss_mask])
            if self.cfg['human_lights']:
                human_lights, human_weights = self.get_human_light(points[miss_mask], directions[miss_mask], human_poses[miss_mask])
            else:
                human_lights, human_weights = torch.zeros_like(outer_lights), torch.zeros(outer_lights.shape[0], 1)
            lights[miss_mask] = outer_lights * (1 - human_weights) + human_lights * human_weights

        if torch.sum(hit_mask)>0:
            lights[hit_mask] = self.get_inner_lights(inters[hit_mask], -directions[hit_mask], normals[hit_mask])

        near_mask = (depth>eps).float()
        lights = lights * near_mask # very near surface does not bring lights
        return lights, human_lights * human_weights, inters, normals, hit_mask

    def fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0)**5.0

    def fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = (view_dirs + directions) # [pn,sn0,3]
        H = F.normalize(H, dim=-1)
        HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0) # [pn,sn0,1]
        fresnel = self.fresnel_schlick(F0, HoV) # [pn,sn0,1]
        return fresnel, H, HoV

    def geometry_schlick_ggx(self, NoV, roughness):
        a = roughness # a = roughness**2: we assume the predicted roughness is already squared

        k = a / 2
        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def geometry_schlick(self, NoV, NoL, roughness):
        ggx2 = self.geometry_schlick_ggx(NoV, roughness)
        ggx1 = self.geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
        def fun(alpha2, cos_theta):
            # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
            cos_theta2 = cos_theta**2
            tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
            return 0.5 * torch.sqrt(1+alpha2*tan_theta2) - 0.5

        alpha_sq = roughness ** 2
        return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))

    def predict_materials(self, pts):
        feats = self.feats_network(pts)
        metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
        roughness = self.roughness_predictor(torch.cat([feats, pts], -1))
        rmax, rmin = 1.0, 0.04**2
        roughness = roughness * (rmax - rmin) + rmin
        albedo = self.albedo_predictor(torch.cat([feats, pts], -1))
        return metallic, roughness, albedo

    def distribution_ggx(self, NoH, roughness):
        a = roughness
        a2 = a**2
        NoH2 = NoH**2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom**2 + 1e-4)

    def geometry(self,NoV, NoL, roughness):
        if self.cfg['geometry_type']=='schlick':
            geometry = self.geometry_schlick(NoV, NoL, roughness)
        elif self.cfg['geometry_type']=='ggx_smith':
            geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
        else:
            raise NotImplementedError
        return geometry

    def shade_mixed(self, pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train):
        F0 = 0.04 * (1 - metallic) + metallic * albedo # [pn,1]

        # sample diffuse directions
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)  # [pn,sn0,3]
        point_num, diffuse_num, _ = diffuse_directions.shape
        # sample specular directions
        specular_directions = self.sample_specular_directions(reflections, roughness, is_train) # [pn,sn1,3]
        specular_num = specular_directions.shape[1]

        # diffuse sample prob
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1))
        diffuse_probability = NoL_d / np.pi * (diffuse_num / (specular_num+diffuse_num))

        # specualr sample prob
        H_s = (view_dirs.unsqueeze(1) + specular_directions) # [pn,sn0,3]
        H_s = F.normalize(H_s, dim=-1)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(view_dirs.unsqueeze(1),H_s)
        specular_probability = self.distribution_ggx(NoH_s, roughness.unsqueeze(1)) * NoH_s / (4 * VoH_s + 1e-5) * (specular_num / (specular_num+diffuse_num)) # D * NoH / (4 * VoH)

        # combine
        directions = torch.cat([diffuse_directions, specular_directions], 1)
        probability = torch.cat([diffuse_probability, specular_probability], 1)
        sn = diffuse_num+specular_num

        # specular
        fresnel, H, HoV = self.fresnel_schlick_directions(F0.unsqueeze(1), view_dirs.unsqueeze(1), directions)
        NoV = saturate_dot(normals, view_dirs).unsqueeze(1) # pn,1,3
        NoL = saturate_dot(normals.unsqueeze(1), directions) # pn,sn,3
        geometry = self.geometry(NoV, NoL, roughness.unsqueeze(1))
        NoH = saturate_dot(normals.unsqueeze(1), H)
        distribution = self.distribution_ggx(NoH, roughness.unsqueeze(1))
        human_poses = human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
        pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
        lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses) # pn,sn,3
        specular_weights = distribution * geometry / (4 * NoV * probability + 1e-5)
        specular_lights =  lights * specular_weights
        specular_colors = torch.mean(fresnel * specular_lights, 1)
        specular_weights = specular_weights * fresnel

        # diffuse only consider diffuse directions
        kd = (1 - metallic.unsqueeze(1))
        diffuse_lights = lights[:,:diffuse_num]
        diffuse_colors = albedo.unsqueeze(1) * kd[:,:diffuse_num] * diffuse_lights
        diffuse_colors = torch.mean(diffuse_colors, 1)

        colors = diffuse_colors + specular_colors
        colors = linear_to_srgb(colors)
        visibility = 1. - torch.mean(light_pts_mask.float(), dim=1).unsqueeze(-1)                     # pn, 1
        indirect_light = torch.mean(lights * light_pts_mask[..., None].float(), dim=1)                # pn, 3
        
        outputs={}
        outputs['albedo'] = albedo
        outputs['roughness'] = roughness
        outputs['metallic'] = metallic
        outputs['human_lights'] = hl.reshape(-1,3)
        outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)),min=0,max=1)
        outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(specular_lights, dim=1)),min=0,max=1)
        diffuse_colors = torch.clamp(linear_to_srgb(diffuse_colors),min=0,max=1)
        specular_colors = torch.clamp(linear_to_srgb(specular_colors),min=0,max=1)
        outputs['diffuse_color'] = diffuse_colors
        outputs['specular_color'] = specular_colors
        outputs['approximate_light'] = torch.clamp(linear_to_srgb(torch.mean(kd[:,:diffuse_num] * diffuse_lights, dim=1)+specular_colors),min=0,max=1)
        outputs['visibility'] = visibility
        outputs['indirect_light'] = indirect_light
        return colors, outputs

    def forward(self, pts, view_dirs, normals, human_poses, step, is_train):
        view_dirs, normals = F.normalize(view_dirs, dim=-1), F.normalize(normals, dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        metallic, roughness, albedo = self.predict_materials(pts)  # [pn,1] [pn,1] [pn,3]
        return self.shade_mixed(pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train)

    def env_light(self, h, w, gamma=True):
        azs = torch.linspace(1.0, 0.0, w) * np.pi*2 - np.pi/2
        els = torch.linspace(1.0, -1.0, h) * np.pi/2

        els, azs = torch.meshgrid(els, azs)
        if self.cfg['is_real']:
            x = torch.cos(els) * torch.cos(azs)
            y = torch.cos(els) * torch.sin(azs)
            z = torch.sin(els)
        else:
            z = torch.cos(els) * torch.cos(azs)
            x = torch.cos(els) * torch.sin(azs)
            y = torch.sin(els)
        xyzs = torch.stack([x,y,z], -1) # h,w,3
        xyzs = xyzs.reshape(h*w,3)
        # xyzs = xyzs @ torch.from_numpy(np.asarray([[0,0,1],[0,1,0],[-1,0,0]],np.float32)).cuda()

        batch_size = 8192
        lights = []
        for ri in range(0, h*w, batch_size):
            with torch.no_grad():
                light = self.predict_outer_lights_pts(xyzs[ri:ri+batch_size])
            lights.append(light)
        if gamma:
            lights = linear_to_srgb(torch.cat(lights, 0)).reshape(h,w,3)
        else:
            lights = (torch.cat(lights, 0)).reshape(h, w, 3)
        return lights

    def predict_outer_lights_pts(self,pts):
        if self.cfg['outer_light_version']=='direction':
            return self.outer_light(self.sph_enc(pts, 0))
        elif self.cfg['outer_light_version']=='sphere_direction':
            return self.outer_light(torch.cat([self.sph_enc(pts, 0),self.sph_enc(pts, 0)],-1))
        else:
            raise NotImplementedError


    def get_env_light(self):
        return self.predict_outer_lights_pts(self.light_pts)

    def material_regularization(self, pts, normals, metallic, roughness, albedo, step):
        # metallic, roughness, albedo = self.predict_materials(pts)
        reg = 0

        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1)
            x = self.get_orthogonal_directions(normals)
            y = torch.cross(normals,x)
            ang = torch.rand(pts.shape[0],1)*np.pi*2
            if self.cfg['change_type']=='constant':
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg['change_eps']
            elif self.cfg['change_type']=='gaussian':
                eps = torch.normal(mean=0.0, std=self.cfg['change_eps'], size=[x.shape[0], 1])
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            m0, r0, a0 = self.predict_materials(pts+change)
            reg = reg + torch.mean((torch.abs(m0-metallic) + torch.abs(r0-roughness) + torch.abs(a0 - albedo)) * self.cfg['reg_lambda1'], dim=1)

        if self.cfg['reg_min_max'] and step is not None and step < 2000:
            # sometimes the roughness and metallic saturate with the sigmoid activation in the early stage
            reg = reg + torch.sum(torch.clamp(roughness - 0.98**2, min=0))
            reg = reg + torch.sum(torch.clamp(0.02**2 - roughness, min=0))
            reg = reg + torch.sum(torch.clamp(metallic - 0.98, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 - metallic, min=0))

        return reg