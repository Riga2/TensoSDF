import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import argparse
from utils.base_utils import load_cfg
import torch
import numpy as np
from tqdm import tqdm
import imageio

import cv2

from skimage.io import imread, imsave
from lpips import LPIPS
from kornia.losses import ssim_loss
_lpips = None

def rgb_to_srgb(f: np.ndarray):
    # f is loaded from .exr
    # output is NOT clipped to [0, 1]
    assert len(f.shape) == 3, f.shape
    assert f.shape[2] == 3, f.shape
    f = np.where(f > 0.0031308, np.power(np.maximum(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * f)
    return f


def srgb_to_rgb(f: np.ndarray):
    # f is LDR
    assert len(f.shape) == 3, f.shape
    assert f.shape[2] == 3, f.shape
    f = np.where(f <= 0.04045, f / 12.92, np.power((np.maximum(f, 0.04045) + 0.055) / 1.055, 2.4))
    return f

def erode_mask(mask, target_size):
    if mask.ndim == 3:
        mask = mask[...,0]
    if mask.dtype == np.float32:
        mask = (mask*255).clip(0, 255).astype(np.uint8)
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    if target_size is not None:
        mask = cv2.resize(mask, (target_size, target_size))
    return (mask > 127).astype(np.float32)

def lpips(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if LPIPS is None:
        return np.nan
    global _lpips
    if _lpips is None:
        _lpips = LPIPS(net='vgg', verbose=False).cuda()
    inputs = rgb_to_srgb(inputs)
    target = rgb_to_srgb(target)

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    return _lpips(inputs, target, normalize=True).item()

def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)


def calc_PSNR(img_pred, img_gt, mask_gt,
              max_value, tonemapping, scale_invariant,
              divide_mask=True):
    # make sure img_pred, img_gt are linear
    '''
        calculate the PSNR between the predicted image and ground truth image.
        a scale is optimized to get best possible PSNR.
        images are clip by max_value_ratio.
        params:
        img_pred: numpy.ndarray of shape [H, W, 3]. predicted HDR image.
        img_gt: numpy.ndarray of shape [H, W, 3]. ground truth HDR image.
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
        max_value: Float. the maximum value of the ground truth image clipped to.
            This is designed to prevent the result being affected by too bright pixels.
        tonemapping: Bool. Whether the images are tone-mapped before comparion.
        divide_mask: Bool. Whether the mse is divided by the foreground area.
    '''
    if mask_gt.ndim == 3:
        mask_gt = mask_gt[..., 0]
    if mask_gt.dtype == np.float32:
        mask_gt = (mask_gt * 255).clip(0, 255).astype(np.uint8)
    else:
        import ipdb; ipdb.set_trace()
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask_gt = cv2.erode(mask_gt, kernel)
    mask_gt = (mask_gt > 127).astype(np.float32)

    img_pred = img_pred * mask_gt[..., None]
    img_gt = img_gt * mask_gt[..., None]
    img_gt[img_gt < 0] = 0

    if scale_invariant:
        img_pred_pixels = img_pred[np.where(mask_gt > 0.5)]
        img_gt_pixels = img_gt[np.where(mask_gt > 0.5)]
        for c in range(3):
            if (img_pred_pixels[:, c] ** 2).sum() <= 1e-6:
                img_pred_pixels[:, c] = np.ones_like(img_pred_pixels[:, c])
                # import ipdb; ipdb.set_trace()
        scale = (img_gt_pixels * img_pred_pixels).sum(axis=0) / (img_pred_pixels ** 2).sum(axis=0)
        assert scale.shape == (3,), scale.shape
        if (scale < 0).any():
            import ipdb; ipdb.set_trace()
        if (img_pred < 0).any():
            import ipdb; ipdb.set_trace()
        if (img_gt < 0).any():
            import ipdb; ipdb.set_trace()
        img_pred = scale * img_pred
        # if not tonemapping:
    #     imageio.imsave("./rescaled.exr", img_pred)
    #     imageio.imsave("./rescaled_gt.exr", img_gt)

    # clip the prediction and the gt img by the maximum_value
    img_pred = np.clip(img_pred, 0, max_value)
    img_gt = np.clip(img_gt, 0, max_value)

    if tonemapping:
        img_pred = rgb_to_srgb(img_pred)
        img_gt = rgb_to_srgb(img_gt)
        # imageio.imsave("./rescaled.png", (img_pred*255).clip(0,255).astype(np.uint8))
        # imageio.imsave("./rescaled_gt.png", (img_gt*255).clip(0,255).astype(np.uint8))

    if not divide_mask:
        mse = ((img_pred - img_gt) ** 2).mean()
        lb = ((np.ones_like(img_gt) * .5 * mask_gt[:, :, None] - img_gt) ** 2).mean()
    else:
        mse = ((img_pred - img_gt) ** 2).sum() / mask_gt.sum()
        lb = ((np.ones_like(img_gt) * .5 * mask_gt[:, :, None] - img_gt) ** 2).sum() / mask_gt.sum()
    out = mse_to_psnr(mse)
    lb = mse_to_psnr(lb)
    out = max(out, lb)
    return out, img_pred, img_gt


def ssim(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if ssim_loss is None:
        return np.nan

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    # image_pred and image_gt: (1, 3, H, W) in range [0, 1]
    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    dssim_ = ssim_loss(inputs, target, 3).item()  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def load_mask_png(path: str):
    f = imread(path).astype(np.float32)
    f = f / 255.
    assert len(f.shape) == 2, f.shape
    return f

def img_read_rgba(path):
    im = imread(path).astype(np.float32)
    im_rgb = np.array(im)[..., :3] / 255.
    im_alpha = np.array(im)[..., [-1]] / 255.
    im = (im_rgb * im_alpha + (1 - im_alpha))
    return im

def img_read_rgb(path):
    im = imread(path).astype(np.float32)
    im_rgb = np.array(im)[..., :3] / 255.
    return im_rgb

def eval_relight(relight_dir, gt_dir):
    relight_env_name = gt_dir.split('/')[-1]
    save_dir = f'data/relight/orb/afterScale/{relight_env_name}'
    os.makedirs(save_dir, exist_ok=True)
    avg_PSNR, avg_SSIM, avg_LPIPS = 0, 0, 0
    msg = ''
    num = len(os.listdir(relight_dir))
    for name in tqdm(os.listdir(relight_dir)):
        relit_path = os.path.join(relight_dir, name)
        mask_path = os.path.join(gt_dir, 'test_mask', name)
        gt_path = os.path.join(gt_dir, 'test', name)
        relit_im = img_read_rgba(relit_path)
        gt_im = img_read_rgb(gt_path)
        mask = load_mask_png(mask_path)
 
        cur_psnr, img_pred, img_gt = calc_PSNR(relit_im, gt_im, mask, max_value=1, tonemapping=False, divide_mask=False, scale_invariant=True)
        cur_lpips = lpips(img_pred, img_gt, mask)
        cur_ssim = ssim(img_gt, img_pred, mask)   
        
        msg += f'{name}, psnr: {cur_psnr}, ssim: {cur_ssim}, lpips: {cur_lpips}\n'
        avg_PSNR += cur_psnr
        avg_SSIM += cur_ssim
        avg_LPIPS += cur_lpips
        
        img_pred_save = np.concatenate([img_pred, mask[..., None]], axis=-1)
        img_gt_save = np.concatenate([img_gt, mask[..., None]], axis=-1)
        
        imageio.imsave(f"{save_dir}/{name}", (img_pred_save*255).clip(0,255).astype(np.uint8))
        imageio.imsave(f"{save_dir}/gt_{name}", (img_gt_save*255).clip(0,255).astype(np.uint8))        
    
    avg_PSNR /= num
    avg_SSIM /= num
    avg_LPIPS /= num
    msg += f'Avg_psnr: {avg_PSNR}, avg_ssim: {avg_SSIM}, avg_lpips: {avg_LPIPS}\n'
    with open(f'{save_dir}/metrics_record.txt', 'a') as f:
        f.write(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--relight_dir', type=str, default='data/relight/orb/noScale/cactus_scene001_relighting_cactus_scene007',
                        help='relighted images from blender')
    parser.add_argument('--gt_dir', type=str, default='/home/riga/NeRF/nerf_data/blender_LDR/cactus_scene007',
                        help='ground truth relighting images from orb LDR datasets')

    flags = parser.parse_args()
    
    eval_relight(flags.relight_dir, flags.gt_dir)