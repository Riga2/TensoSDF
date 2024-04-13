import numpy as np
import torch

class Loss:
    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class NeRFRenderLoss(Loss):
    default_cfg={
        'radiance_init_rgb_weight': 0.0,
        'radiance_final_rgb_weight': 0.0,
        'radiance_weight_anneal_begin': 0,
        'radiance_weight_anneal_end': 0,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def get_radiance_weight(self, step):
        if step < self.cfg['radiance_weight_anneal_begin']:
            return self.cfg['radiance_init_rgb_weight']
        elif self.cfg['radiance_weight_anneal_begin'] <= step < self.cfg['radiance_weight_anneal_end']:
            lerp_ratio = (step - self.cfg['radiance_weight_anneal_begin']) / (self.cfg['radiance_weight_anneal_end'] - self.cfg['radiance_weight_anneal_begin'])
            return self.cfg['radiance_init_rgb_weight'] * (1.0 - lerp_ratio) + self.cfg['radiance_final_rgb_weight'] * lerp_ratio
        else:
            return self.cfg['radiance_final_rgb_weight']

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        # radiance_loss_weight = self.get_radiance_weight(step)
        # inverse_loss_weight = 1.0 - radiance_loss_weight
        radiance_loss_weight = 1.0
        inverse_loss_weight = 1.0
        if 'loss_rgb' in data_pr: outputs['loss_rgb']=data_pr['loss_rgb'] * inverse_loss_weight # self.cfg['inverse_rgb_weight']
        if 'loss_radiance' in data_pr: outputs['loss_radiance']=data_pr['loss_radiance'] * radiance_loss_weight # self.cfg['radiance_field_rgb_weight']
        if 'loss_rgb_fine' in data_pr: outputs['loss_rgb_fine']=data_pr['loss_rgb_fine']
        if 'loss_global_rgb' in data_pr: outputs['loss_global_rgb']=data_pr['loss_global_rgb']
        if 'loss_rgb_inner' in data_pr: outputs['loss_rgb_inner']=data_pr['loss_rgb_inner']
        if 'loss_rgb0' in data_pr: outputs['loss_rgb0']=data_pr['loss_rgb0']
        if 'loss_rgb1' in data_pr: outputs['loss_rgb1']=data_pr['loss_rgb1']
        # if 'loss_masks' in data_pr: outputs['loss_masks']=data_pr['loss_masks']
        return outputs

class EikonalLoss(Loss):
    default_cfg={
        "eikonal_weight": 0.1,
        'eikonal_weight_anneal_begin': 0,
        'eikonal_weight_anneal_end': 0,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg, **cfg}

    def get_eikonal_weight(self, step):
        if step<self.cfg['eikonal_weight_anneal_begin']:
            return 0.0
        elif self.cfg['eikonal_weight_anneal_begin'] <= step < self.cfg['eikonal_weight_anneal_end']:
            return self.cfg['eikonal_weight'] * (step - self.cfg['eikonal_weight_anneal_begin']) / \
                   (self.cfg['eikonal_weight_anneal_end']-self.cfg['eikonal_weight_anneal_begin'])
        else:
            return self.cfg['eikonal_weight']

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        weight = self.get_eikonal_weight(step)
        outputs={'loss_eikonal': data_pr['gradient_error'] * weight}
        return outputs

class TV_Loss(Loss):
    default_cfg={
        'TV_weight_sdf' : 0.1,
        'TV_weight_app' : 0.1,
        'TV_weight_light' : 0.1,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_tv_sdf' in data_pr:
            outputs['loss_tv_sdf'] = torch.mean(data_pr['loss_tv_sdf']).reshape(1) * self.cfg['TV_weight_sdf']
        if 'loss_tv_app' in data_pr:
            outputs['loss_tv_app'] = torch.mean(data_pr['loss_tv_app']).reshape(1) * self.cfg['TV_weight_app']
        if 'loss_tv_light' in data_pr:
            outputs['loss_tv_light'] = torch.mean(data_pr['loss_tv_light']).reshape(1) * self.cfg['TV_weight_light']
        return outputs
    
class Sparse_Loss(Loss):
    default_cfg={
        'sparse_weight' : 0.02,
        'sparse_ratio': [1.0, 1.0]
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.init_sparse_weight = self.cfg['sparse_weight']
        self.cur_ratio = 1.0

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_sparse' in data_pr:
            for i in range(len(self.cfg['sparse_update_list'])-1, 0, -1):
                if step >= self.cfg['sparse_update_list'][i]:
                    self.cur_ratio = self.cfg['sparse_ratio'][i]
                    break
            outputs['loss_sparse'] = data_pr['loss_sparse'] * self.init_sparse_weight * self.cur_ratio
        return outputs

class Hessian_Loss(Loss):
    default_cfg={
        'hessian_weight' : 5e-4
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.init_hessian_weight = self.cfg['hessian_weight']
        self.cur_ratio = 1.0

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_hessian' in data_pr:
            for i in range(len(self.cfg['hessian_update_list'])-1, 0, -1):
                if step >= self.cfg['hessian_update_list'][i]:
                    self.cur_ratio = self.cfg['hessian_ratio'][i]
                    break
            outputs['loss_hessian'] = data_pr['loss_hessian'] * self.init_hessian_weight * self.cur_ratio
        return outputs

class MaterialRegLoss(Loss):
    default_cfg={
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_mat_reg' in data_pr: outputs['loss_mat_reg'] = data_pr['loss_mat_reg']
        if 'loss_diffuse_light' in data_pr: outputs['loss_diffuse_light'] = data_pr['loss_diffuse_light']
        return outputs

class StdRecorder(Loss):
    default_cfg={
        'apply_std_loss': False,
        'std_loss_weight': 0.05,
        'std_loss_weight_type': 'constant',
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'std' in data_pr:
            outputs['std'] = data_pr['std']
            if self.cfg['apply_std_loss']:
                if self.cfg['std_loss_weight_type'] =='constant':
                    outputs['loss_std'] = data_pr['std']*self.cfg['std_loss_weight']
                else:
                    raise NotImplementedError
        if 'inner_std' in data_pr: outputs['inner_std'] = data_pr['inner_std']
        if 'outer_std' in data_pr: outputs['outer_std'] = data_pr['outer_std']
        return outputs

class OccLoss(Loss):
    default_cfg={}
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_occ' in data_pr:
            outputs['loss_occ'] = torch.mean(data_pr['loss_occ']).reshape(1)
        return outputs

class InitSDFRegLoss(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        reg_step = 1000
        small_threshold = 0.1
        large_threshold = 1.05
        if 'sdf_vals' in data_pr and 'sdf_pts' in data_pr and step<reg_step:
            norm = torch.norm(data_pr['sdf_pts'], dim=-1)
            sdf = data_pr['sdf_vals']
            small_mask = norm<small_threshold
            if torch.sum(small_mask)>0:
                bounds = norm[small_mask] - small_threshold # 0-small_threshold -> 0
                # we want sdf - bounds < 0
                small_loss = torch.mean(torch.clamp(sdf[small_mask] - bounds, min=0.0))
                small_loss = torch.sum(small_loss) / (torch.sum(small_loss > 1e-5) + 1e-3)
            else:
                small_loss = torch.zeros(1)

            large_mask = norm > large_threshold
            if torch.sum(large_mask)>0:
                bounds = norm[large_mask] - large_threshold # 0 -> 1 - large_threshold
                # we want sdf - bounds > 0 => bounds - sdf < 0
                large_loss = torch.clamp(bounds - sdf[large_mask], min=0.0)
                large_loss = torch.sum(large_loss) / (torch.sum(large_loss > 1e-5) + 1e-3)
            else:
                large_loss = torch.zeros(1)

            anneal_weights = (np.cos((step / reg_step) * np.pi) + 1) / 2
            return {'loss_sdf_large': large_loss*anneal_weights, 'loss_sdf_small': small_loss*anneal_weights}
        else:
            return {}


class Gaussian_Loss(Loss):
    default_cfg={
        'gaussian_weight' : 5e-4
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_gaussian' in data_pr:
            outputs['loss_gaussian'] = data_pr['loss_gaussian'] * self.cfg['gaussian_weight']
        return outputs

class Predicted_Normal_Loss(Loss):
    default_cfg={
        'normal_diff_weight' : 0.05
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'normals_pred_error' in data_pr:
            outputs['loss_diff'] = data_pr['normals_pred_error'] * self.cfg['normal_diff_weight']
        else:
            outputs['loss_diff'] = torch.zeros(1)
        return outputs    
    
class Denoiser_Loss(Loss):
    default_cfg={
        'denoiser_weight' : 0.02
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs={}
        if 'loss_denoising' in data_pr:
            outputs['loss_denoising'] = data_pr['loss_denoising'] * self.cfg['denoiser_weight']
        return outputs

class MaskLoss(Loss):
    default_cfg = {
        'mask_loss_weight': 0.01,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_mask' in data_pr:
            outputs['loss_mask'] = data_pr['loss_mask'].reshape(1) * self.cfg['mask_loss_weight']
        return outputs

name2loss={
    'nerf_render': NeRFRenderLoss,
    'eikonal': EikonalLoss,
    'std': StdRecorder,
    'init_sdf_reg': InitSDFRegLoss,
    'occ': OccLoss,
    'mask': MaskLoss,

    'mat_reg': MaterialRegLoss,
    'TV' : TV_Loss,
    'Sparse' : Sparse_Loss,
    'Hessian': Hessian_Loss,
    'Gaussian': Gaussian_Loss,
    'Pred_Normal': Predicted_Normal_Loss,
    'Normal_Denoiser': Denoiser_Loss,
}