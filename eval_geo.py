import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random
import argparse
from utils.base_utils import load_cfg
import torch
import numpy as np
from tqdm import tqdm

from network.invRenderer import name2renderer
from dataset.database import parse_database_name
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from utils.base_utils import color_map_backward, color_map_forward, rgb_lpips
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network.other_field import GaussianBlur2D, GaussianBlur1D
import torch.nn.functional as F
from skimage.io import imread, imsave

class ShapeTester:
    default_cfg={
        "multi_gpus": False,
        "worker_num": 8,
        'random_seed': 6033,
        'isBGWhite': True,
    }
    
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name=cfg['name']
        self.model_dir=os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir):
            raise ImportError
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')
        self.base_save_path = os.path.join('data/nvs', cfg['name'])
        os.makedirs(self.base_save_path, exist_ok=True)

    def _init_dataset(self):
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'], isTest=True, isWhiteBG=self.cfg['isBGWhite'])
        self.test_ids = self.database.get_img_ids()
        self.dataloader = DataLoader(self.test_ids, 1, False, num_workers=self.cfg['worker_num'])
        print(f'Test set len {len(self.test_ids)}')

    def _init_network(self):
        best_para,start_step=0,0
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            if 'kwargs' in checkpoint:
                kwargs = checkpoint['kwargs']
                self.cfg.update(kwargs)
            self.network = name2renderer[self.cfg['network']](self.cfg).cuda()
            self.network.load_ckpt(checkpoint)
            print(f'==> resuming from step {start_step} best para {best_para}')
        else:
            raise NotImplementedError

    def run(self):
        self._init_dataset()
        self._init_network()

        num = len(self.test_ids)
        img_avg_psnr = 0
        img_avg_ssim = 0
        normal_avg_mae = 0
        def dir_maker(name):
            dir = os.path.join(self.base_save_path, name)
            os.makedirs(dir, exist_ok=True)             
            return dir
                        
        imgs_dir = dir_maker('imgs')
        normals_dir = dir_maker('normals')
        normals_vis_dir = dir_maker('normals_vis')
        albedo_dir = dir_maker('albedo')
        rough_dir = dir_maker('roughness')
        occ_dir = dir_maker('occ')
        diff_spec_color_dir = dir_maker('diff_spec_color')
        light_dir = dir_maker('light')
        radiance_dir = dir_maker('radiance')
        imgs_diff_dir = dir_maker('imgs_diff')

        save_func = lambda save_dir, index, im : cv2.imwrite(os.path.join(save_dir, str(int(index)) + '.png'), im[..., ::-1])
        save_exr = lambda save_dir, index, im : cv2.imwrite(os.path.join(save_dir, str(int(index)) + '.exr'), im[..., ::-1])
        per_res_msg = ""
        for _, ind in tqdm(enumerate(self.test_ids)):
            pose = self.database.get_pose(ind)
            K = self.database.get_K(ind)
            gt_imgs = self.database.get_image(ind)
            h, w = gt_imgs.shape[:2]
            with torch.no_grad():
                outputs = self.network.nvs(pose, K, h, w)
            for k in outputs:
                if k != 'normal':
                    outputs[k] = color_map_backward(outputs[k])
            cur_psnr = psnr(gt_imgs, outputs['color'])
            cur_ssim = ssim(gt_imgs, outputs['color'], win_size=11, channel_axis=2, data_range=255)  
            img_avg_psnr += cur_psnr
            img_avg_ssim += cur_ssim
            per_res_msg += f'{ind:03} psnr: {cur_psnr}, ssim: {cur_ssim}'
            
            gt_normals = self.database.get_normal(ind)
            gt_normals = normalize_numpy(gt_normals)
            cur_mae = np.mean(np.arccos(np.clip(np.sum(gt_normals * outputs['normal'], axis=-1), -1, 1)) * 180 / np.pi)
            normal_avg_mae += cur_mae
            per_res_msg += f', mae: {cur_mae}'
            
            normal_rgb = color_map_backward((outputs['normal'] + 1.0) * 0.5)
            gt_normal_rgb = color_map_backward((gt_normals + 1.0) * 0.5)
            normal_diff_map = np.repeat(color_map_backward(np.sum(np.power(outputs['normal'] - gt_normals, 2), axis=-1, keepdims=True)), 3, axis=-1)
            save_func(normals_dir, ind, np.concatenate([normal_rgb, gt_normal_rgb, normal_diff_map], axis=1))

            per_res_msg += '\n'
            color_diff = np.repeat(np.clip(np.sum(np.abs(outputs['color'].astype(np.float32) - gt_imgs.astype(np.float32)), axis=-1, keepdims=True), a_min=0, a_max=255.0).astype(np.uint8), 3, axis=-1)
            save_func(imgs_diff_dir, ind, color_diff)
            save_func(imgs_dir, ind, outputs['color'])
            save_func(albedo_dir, ind, outputs['albedo'])
            save_func(rough_dir, ind, outputs['roughness'])
            save_func(radiance_dir, ind, outputs['radiance'])
            save_func(occ_dir, ind, np.concatenate((outputs['occ_predict'], outputs['occ_trace']), axis=1))
            save_func(diff_spec_color_dir, ind, np.concatenate((outputs['diff_color'], outputs['spec_color']), axis=1))
            save_func(light_dir, ind, np.concatenate((outputs['diff_light'], outputs['spec_light'], outputs['indirect_light']), axis=1))
            save_func(normals_vis_dir, ind, outputs['normal_vis'])

        img_avg_psnr /= num
        img_avg_ssim /= num
        normal_avg_mae /= num

        saved_message = f'{self.model_name}: \n' \
                        + f'\tPSNR_nvs: {img_avg_psnr:.3f}, SSIM_nvs: {img_avg_ssim:.5f}' \
                        + f', Normal_MAE_nvs: {normal_avg_mae:.5f}'
        with open(f'{self.base_save_path}/metrics_record.txt', 'a') as f:
            f.write(saved_message)   
        print(saved_message)
        with open(f'{self.base_save_path}/per_record.txt', 'a') as f:
            f.write(per_res_msg)   


def normalize_numpy(x, axis=-1, order=2):
    norm = np.linalg.norm(x, ord=order, axis=axis, keepdims=True)
    return x / np.maximum(norm, np.finfo(float).eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/shape/syn/motor.yaml')
    flags = parser.parse_args()
    ShapeTester(load_cfg(flags.cfg)).run()