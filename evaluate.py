import os
import glob

import numpy as np
import torch
from tqdm import tqdm
from skimage import io
import pyiqa

def img2tensor(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.ndim == 2:
        img = img[:, :, None]

    img = img.astype(np.float32)

    vmin = img.min()
    vmax = img.max()

    if vmax <= 1.0 + 1e-6:
        img_norm = img
    else:
        if vmax <= 255.0 * 1.5:
            img_norm = img / 255.0
        elif vmax <= 65535.0 * 1.5:
            img_norm = img / 65535.0
        else:
            img_norm = (img - vmin) / (vmax - vmin + 1e-6)

    img_norm = img_norm.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_norm)
    return tensor.unsqueeze(0)


def rgb2ycbcr_pt(img, y_only=True):
    if img.ndim == 3:
        img = img.unsqueeze(0)

    assert img.dim() == 4, f"img must be 4D, got {img.shape}"

    if img.size(1) == 1:
        img = img.repeat(1, 3, 1, 1)

    r = img[:, 0:1, :, :]
    g = img[:, 1:2, :, :]
    b = img[:, 2:3, :, :]

    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128.0 / 255.0
    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128.0 / 255.0

    if y_only:
        return y
    return torch.cat([y, cb, cr], dim=1)

def metrics(hr_dir, sr_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)

    def collect_images(folder):
        paths = []
        paths.extend(sorted(glob.glob(os.path.join(folder, '*.JPEG'))))
        paths.extend(sorted(glob.glob(os.path.join(folder, '*.jpg'))))
        paths.extend(sorted(glob.glob(os.path.join(folder, '*.png'))))
        paths.extend(sorted(glob.glob(os.path.join(folder, '*.bmp'))))
        return paths

    gt_img_paths = collect_images(hr_dir)
    sr_img_paths = collect_images(sr_dir)

    if len(gt_img_paths) == 0:
        raise ValueError(f"No images found in HR directory {hr_dir}!")
    if len(sr_img_paths) == 0:
        raise ValueError(f"No images found in SR directory {sr_dir}!")

    if len(gt_img_paths) != len(sr_img_paths):
        print(f"Warning: The number of HR images ({len(gt_img_paths)}) does not match the number of SR images ({len(sr_img_paths)}). Pairing will be done based on the smaller count.")

    pair_num = min(len(gt_img_paths), len(sr_img_paths))

    print(f"Computing metrics for the first {pair_num} image pairs in order.")
    print(f"Evaluating {sr_dir}:")

    psnr_folder = []
    ssim_folder = []
    dists_score = []
    lpips_iqa = []
    musiq_iqa = []
    maniqa_iqa = []

    for i in tqdm(range(pair_num)):
        gt_img_path = gt_img_paths[i]
        sr_img_path = sr_img_paths[i]

        gt_np = io.imread(gt_img_path)
        sr_np = io.imread(sr_img_path)

        img1 = rgb2ycbcr_pt(img2tensor(gt_np), y_only=True).to(device=device, dtype=torch.float32)
        img2 = rgb2ycbcr_pt(img2tensor(sr_np), y_only=True).to(device=device, dtype=torch.float32)

        ssim_folder.append(ssim_metric(img1, img2))
        psnr_folder.append(psnr_metric(img1, img2))

        lpips_iqa.append(lpips_iqa_metric(sr_img_path, gt_img_path))
        musiq_iqa.append(musiq_iqa_metric(sr_img_path))
        maniqa_iqa.append(maniqa_metric(sr_img_path))
        dists_score.append(dists_iqa_metric(sr_img_path, gt_img_path))

    m_psnr = sum(psnr_folder) / len(psnr_folder)
    m_ssim = sum(ssim_folder) / len(ssim_folder)
    m_lpips = sum(lpips_iqa) / len(lpips_iqa)
    m_dists = sum(dists_score) / len(dists_score)
    musiq_mean = sum(musiq_iqa) / len(musiq_iqa)
    maniqa_mean = sum(maniqa_iqa) / len(maniqa_iqa)

    print(f"PSNR = {m_psnr}")
    print(f"SSIM = {m_ssim}")
    print(f"LPIPS = {m_lpips.item() if torch.is_tensor(m_lpips) else m_lpips}")
    print(f"DISTS = {m_dists}")
    print(f"MUSIQ = {musiq_mean.item() if torch.is_tensor(musiq_mean) else musiq_mean}")
    print(f"MANIQA = {maniqa_mean.item() if torch.is_tensor(maniqa_mean) else maniqa_mean}")

    m_psnr_val = m_psnr.item() if torch.is_tensor(m_psnr) else float(m_psnr)
    m_ssim_val = m_ssim.item() if torch.is_tensor(m_ssim) else float(m_ssim)
    m_lpips_val = m_lpips.item() if torch.is_tensor(m_lpips) else float(m_lpips)
    m_dists_val = m_dists.item() if torch.is_tensor(m_dists) else float(m_dists)
    musiq_val = musiq_mean.item() if torch.is_tensor(musiq_mean) else float(musiq_mean)
    maniqa_val = maniqa_mean.item() if torch.is_tensor(maniqa_mean) else float(maniqa_mean)

    return (
        m_psnr_val,
        m_ssim_val,
        m_lpips_val,
        m_dists_val,
        musiq_val,
        maniqa_val
    )


if __name__ == "__main__":
    hr_folders = [
        "testset/HR",
    ]
    
    sr_folders = [
        "testset/pred",
    ]
    

    for hr_folder, sr_folder in zip(hr_folders, sr_folders):
        metrics(hr_folder, sr_folder)
