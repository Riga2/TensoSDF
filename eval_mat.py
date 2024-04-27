import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from utils.raw_utils import linear_to_srgb
import subprocess

class MaterialTester:
    default_cfg={
        "multi_gpus": False,
        "worker_num": 8,
        'random_seed': 6033,
        'isBGWhite': True,
        'albedoRescale': 0,
    }
    
    def __init__(self, cfg, blender_path, env_dir, orb_settings = None):
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

        self.albedo_rescale = self.cfg['albedoRescale']
        self.blender_path = blender_path
        self.env_dir = env_dir
        self.env_maps = ['bridge', 'city', 'courtyard', 'interior', 'night']
        self.orb_settings = orb_settings

        self._init_dataset()
        self._init_network()

    def _init_dataset(self): 
        self.database_type, self.scene_name = self.cfg['database_name'].split('/')
        self.scene_dir = f'{self.cfg["dataset_dir"]}/{self.scene_name}'
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
            self.network = name2renderer[self.cfg['network']](self.cfg).eval().cuda()
            self.network.load_ckpt(checkpoint)
            self.step = start_step
            print(f'==> resuming from step {start_step} best para {best_para}')
        else:
            raise NotImplementedError

    def calc_albedo_rescale(self):
        os.makedirs(self.base_save_path, exist_ok=True)
        h, w = int(self.database.H), int(self.database.W)
        gt_albedo_lst, pred_albedo_lst = [], []
        rescale_sample_num = 20
        sample_interval = len(self.dataloader) // rescale_sample_num        
        for _, ind in tqdm(enumerate(self.dataloader)):
            if (ind+1) % sample_interval == 0:
                pose = self.database.get_pose(ind)
                K = self.database.get_K(ind)
                with torch.no_grad():
                    outputs = self.network.nvs(pose, K, h, w)

                if self.database_type == 'tensoSDF':
                    # tips: The gt 'albedo' from blender is the albedo * (1-metallic), so we align with it here
                    albedo_pred = outputs['albedo'] * (1. - outputs['metallic'])
                else:
                    albedo_pred = outputs['albedo']
                
                gt_albedo = self.database.get_albedo(ind)                    
                gt_mask = self.database.get_mask(ind) > 0.
                pred_albedo_lst.append(albedo_pred[gt_mask])
                gt_albedo_lst.append(gt_albedo[gt_mask]) 
                
        gt_albedo_samples = np.concatenate(gt_albedo_lst, axis=0)
        pred_albedo_samples = np.concatenate(pred_albedo_lst, axis=0)
        self.single_channel_ratio = np.median((gt_albedo_samples / pred_albedo_samples.clip(min=1e-6))[..., 0])
        self.three_channel_ratio = np.median(gt_albedo_samples / pred_albedo_samples.clip(min=1e-6), axis=0)
        saved_message = f'single channel rescale ratio: {self.single_channel_ratio}, three channels rescale ratio: {self.three_channel_ratio}'                
        with open(f'{self.base_save_path}/albedoRescale_record.txt', 'a') as f:
            f.write(saved_message)
        print(saved_message)


    def extract_materials(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.material_dir = f'data/materials/{self.model_name}-{self.step}'
        os.makedirs(self.material_dir, exist_ok=True)
        with torch.no_grad():
            materials = self.network.predict_materials()
            if self.albedo_rescale != 0:
                self.calc_albedo_rescale()
                if self.albedo_rescale == 1:
                    materials['albedo'] *= self.single_channel_ratio
                elif self.albedo_rescale == 2:
                    materials['albedo'] *= self.three_channel_ratio
                else:
                    raise NotImplementedError
            print(
                'warning!!!!! we transform both albedo/metallic/roughness with gamma correction because our blender script uses vertex colors to store them, '
                'it seems blender will apply an inverse gamma correction so that the results will be incorrect without this gamma correct\n'
                'for more information refer to https://blender.stackexchange.com/questions/87576/vertex-colors-loose-color-data/87583#87583')
            np.save(f'{self.material_dir}/metallic.npy', linear_to_srgb(materials['metallic']))
            np.save(f'{self.material_dir}/roughness.npy', linear_to_srgb(materials['roughness']))
            np.save(f'{self.material_dir}/albedo.npy', linear_to_srgb(materials['albedo']))
        
    def relight(self):
        if self.database_type == 'tensoSDF':
            for env_name in self.env_maps:
                save_dir_name = f'{self.scene_name}_{env_name}'
                relight_save_dir = f'data/relight/{save_dir_name}'
                cmds=[
                    self.blender_path, '--background', '--python', 'blender_backend/relight_backend.py', '--',
                    '--output', relight_save_dir,
                    '--mesh', self.cfg['mesh'],
                    '--material', self.material_dir,
                    '--env_fn', f'{self.env_dir}/{env_name}.exr',
                    '--gt', self.scene_dir,
                    '--env_name', env_name,
                ]
                if self.cfg['trans']:
                    cmds.append('--trans')
                subprocess.run(cmds)
            self.eval_relight_results()
        elif self.database_type == 'orb':
            save_dir_name = f'{self.scene_name}_relighting_{self.orb_settings["orb_relight_env_name"]}'
            relight_save_dir = f'data/relight/orb/noScale/{save_dir_name}'
            env_dir = f'{self.orb_settings["orb_relight_gt_dir"]}/{self.orb_settings["orb_relight_env_name"]}/env_map'
            pose_dir = f'{self.orb_settings["orb_blender_dir"]}/{self.scene_name}'
            cmds=[
                self.blender_path, '--background', '--python', 'blender_backend/relight_backend.py', '--',
                '--output', relight_save_dir,
                '--mesh', self.cfg['mesh'],
                '--material', self.material_dir,
                '--env_fn', env_dir,
                '--gt', pose_dir,
                '--dataset', self.database_type,
                '--env_name', self.orb_settings['orb_relight_env_name'],
            ]
            if self.cfg['trans']:
                cmds.append('--trans')
            subprocess.run(cmds)
        else:
            raise NotImplementedError            
    
    def eval_relight_results(self):
        avg_psnr, avg_ssim, avg_lpips, total_num = 0., 0., 0., 0.
        for env_name in self.env_maps:
            relight_dir = f'data/relight/{self.scene_name}_{env_name}'
            gt_dir = f'{self.scene_dir}/test_relight'
            num = len(os.listdir(relight_dir))
            total_num += num
            
            for i in tqdm(range(num)):
                im_name = f'{i}.png'
                im_path = os.path.join(relight_dir, im_name)
                gt_path = os.path.join(gt_dir, env_name, f'r_{i}.png')
                img = img_read_rgba(im_path)
                gt_img = img_read_rgba(gt_path)
                
                cur_psnr = psnr(gt_img, img)
                cur_ssim = ssim(gt_img, img, win_size=11, channel_axis=2, data_range=255) 
                cur_lpips = rgb_lpips(gt_img, img, 'vgg', 'cuda')
                avg_psnr += cur_psnr
                avg_ssim += cur_ssim
                avg_lpips += cur_lpips
        avg_psnr, avg_ssim, avg_lpips = avg_psnr / total_num, avg_ssim / total_num, avg_lpips / total_num
        print(f'avg_psnr: {avg_psnr}, avg_ssim: {avg_ssim}, avg_lpips: {avg_lpips}')
        msg = f'avg_psnr: {avg_psnr}, avg_ssim: {avg_ssim}, avg_lpips: {avg_lpips}, total_num:{total_num}\n'
        with open(f'data/relight/{self.scene_name}_relight_record.txt', 'a') as f:
            f.write(msg)        
            
    def run_nvs(self):
        os.makedirs(self.base_save_path, exist_ok=True)
        h, w = int(self.database.H), int(self.database.W)
        num = len(self.test_ids)
        img_avg_psnr = 0
        img_avg_ssim = 0
        img_avg_lpips = 0
        def dir_maker(name):
            dir = os.path.join(self.base_save_path, name)
            os.makedirs(dir, exist_ok=True)             
            return dir
                        
        imgs_dir = dir_maker('imgs')
        albedo_dir = dir_maker('albedo')
        metallic_dir = dir_maker('metallic')
        rough_dir = dir_maker('roughness')
        diff_spec_color_dir = dir_maker('diff_spec_color')
        light_dir = dir_maker('light')
        
        save_func = lambda save_dir, index, im : cv2.imwrite(os.path.join(save_dir, str(int(index)) + '.png'), im[..., ::-1])
        for _, ind in tqdm(enumerate(self.dataloader)):
            pose = self.database.get_pose(ind)
            K = self.database.get_K(ind)
            gt_imgs = self.database.get_image(ind)
            with torch.no_grad():
                outputs = self.network.nvs(pose, K, h, w)
            for k in outputs:
                outputs[k] = color_map_backward(outputs[k])
            img_avg_psnr += psnr(gt_imgs, outputs['color'])
            img_avg_ssim += ssim(gt_imgs, outputs['color'], win_size=11, channel_axis=2, data_range=255)
            img_avg_lpips += rgb_lpips(gt_imgs, outputs['color'], 'vgg', 'cuda')

            save_func(imgs_dir, ind, outputs['color'])
            save_func(albedo_dir, ind, outputs['albedo'])
            save_func(rough_dir, ind, outputs['roughness'])
            save_func(metallic_dir, ind, outputs['metallic'])
            save_func(diff_spec_color_dir, ind, np.concatenate((outputs['diff_color'], outputs['spec_color']), axis=1))
            save_func(light_dir, ind, np.concatenate((outputs['diff_light'], outputs['spec_light'], outputs['indirect_light'], outputs['occ_trace'].repeat(3, axis=-1)), axis=1))
            
        img_avg_psnr /= num
        img_avg_ssim /= num
        img_avg_lpips /= num
        saved_message = f'{self.model_name}: \n' \
                        + f'\tPSNR_nvs: {img_avg_psnr:.3f}, SSIM_nvs: {img_avg_ssim:.5f}, LPIPS_nvs: {img_avg_lpips:.5f}'
        with open(f'{self.base_save_path}/metrics_record.txt', 'a') as f:
            f.write(saved_message)
        print(saved_message)


def img_read_rgba(path):
    im = imread(path).astype(np.float32)
    im_rgb = np.array(im)[..., :3] / 255.
    im_alpha = np.array(im)[..., [-1]] / 255.
    im = (im_rgb * im_alpha + (1 - im_alpha))
    return color_map_backward(im)
     
def img_read_rgb(path):
    im = imread(path).astype(np.float32)
    im_rgb = np.array(im)[..., :3] / 255.
    return im_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mat/syn/compressor.yaml')
    parser.add_argument('--blender', type=str, default='/home/riga/blender-3.6.5-linux-x64/blender', 
                        help='the blender path')
    parser.add_argument('--env_dir', type=str, default='/home/riga/NeRF/nerf_data/env_maps/tensoIR/envmaps_1k_exr', 
                        help='env dir')
    #---------------------------Following settings are used in orb datasets-----------------------------------
    parser.add_argument('--orb_relight_gt_dir', type=str, default='/home/riga/NeRF/nerf_data/ground_truth',
                        help='base dir of orb ground-truth')
    parser.add_argument('--orb_relight_env', type=str, default='cactus_scene007',
                        help='orb relight env name')  
    parser.add_argument('--orb_blender_dir', type=str, default='/home/riga/NeRF/nerf_data/blender_LDR',
                        help='base dir of orb ldr blender datasets')
    flags = parser.parse_args()
    
    orb_settings = {
        'orb_relight_env_name': flags.orb_relight_env,
        'orb_relight_gt_dir': flags.orb_relight_gt_dir,
        'orb_blender_dir': flags.orb_blender_dir,
    }
    matTester = MaterialTester(load_cfg(flags.cfg), flags.blender, flags.env_dir, orb_settings)

    matTester.extract_materials()
    
    matTester.relight()