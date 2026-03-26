import os
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
# from tqdm.auto import tqdm

import diffusers
# import utils.misc as misc

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from tadiisr_tile import TAD_IISR
from my_utils.testing_utils import parse_args_paired_testing, PlainDataset, lr_proc
# from utils.util_image import ImageSpliterTh
from my_utils.utils import instantiate_from_config
from pathlib import Path
from utils import util_image
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

from re_de_net import SwinIR
from IDEM import idem

def main(args):
    config = OmegaConf.load(args.base_config)

    if args.pretrained_path is None:
        raise ValueError(
            "This modified inference expects a checkpoint compatible with the modified training. "
            "Please pass --pretrained_path pointing to your saved model_*.pkl."
        )
    pretrained_path = args.pretrained_path

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        if args.redegraded_dir is not None:
            os.makedirs(args.redegraded_dir, exist_ok=True)

    # initialize net_sr
    net_sr = TAD_IISR(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=pretrained_path, args=args)
    net_sr.set_eval()

    # initialize degradation encoder and re-degradation net (match modified training)
    if args.de_enc_path is None or args.re_de_path is None:
        raise ValueError("Please provide --de_enc_path and --re_de_path to run inference with modified training logic.")

    de_enc_net = idem(in_channels=3, out_channels=64).cuda().eval()
    de_enc_net.load_state_dict(torch.load(args.de_enc_path, map_location="cpu"), strict=True)

    re_de_net = SwinIR(
        upscale=2,
        img_size=(512, 512),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='downscale',
        cond_in_chans=64,
        # use_checkpoint=True,
    ).cuda().eval()
    re_de_net.load_state_dict(torch.load(args.re_de_path, map_location="cpu"), strict=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_val = PlainDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_sr, de_enc_net, re_de_net = accelerator.prepare(net_sr, de_enc_net, re_de_net)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    de_enc_net.to(accelerator.device, dtype=weight_dtype)
    re_de_net.to(accelerator.device, dtype=weight_dtype)
      
    offset = args.padding_offset
    for step, batch_val in enumerate(dl_val):
        lr_path = batch_val['lr_path'][0]
        (path, name) = os.path.split(lr_path)

        im_lr = batch_val['lr'].cuda()
        im_lr = im_lr.to(memory_format=torch.contiguous_format).float()    

        ori_h, ori_w = im_lr.shape[2:]
        im_lr_resize = F.interpolate(
            im_lr,
            size=(ori_h * config.sf,
                  ori_w * config.sf),
            mode='bilinear',
            align_corners=False # align_corners with this model causes the output to be shifted, presumably due to training without align_corners
            )

        im_lr_resize = im_lr_resize.contiguous() 
        im_lr_resize_norm = im_lr_resize * 2 - 1.0
        im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]

        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
        
        B = im_lr_resize.size(0)
        with torch.no_grad():
            # forward pass
            deg_score = de_enc_net(im_lr)
            pos_tag_prompt = [args.pos_prompt for _ in range(B)]
            neg_tag_prompt = [args.neg_prompt for _ in range(B)]
            x_tgt_pred = accelerator.unwrap_model(net_sr)(im_lr_resize_norm, deg_score, pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).clamp(0, 1).cpu().detach()

            # re-degrade the SR output back to LR (optional saving)
            re_de_lr = None
            do_save_redegraded = args.save_redegraded or (args.redegraded_dir is not None)
            if do_save_redegraded:
                re_de_lr = re_de_net((x_tgt_pred * 0.5 + 0.5).clamp(0, 1), deg_score)
                re_de_lr = re_de_lr.clamp(0, 1).cpu().detach()

        output_pil = transforms.ToPILImage()(out_img[0])

        if args.align_method == 'nofix':
            output_pil = output_pil
        else:
            im_lr_resize = transforms.ToPILImage()(im_lr_resize[0].cpu().detach())
            if args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(output_pil, im_lr_resize)
            elif args.align_method == 'adain':
                output_pil = adain_color_fix(output_pil, im_lr_resize)

        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname+'.png')
        output_pil.save(outf)

        if (args.save_redegraded or args.redegraded_dir is not None) and re_de_lr is not None:
            rede_pil = transforms.ToPILImage()(re_de_lr[0])
            rede_dir = args.redegraded_dir if args.redegraded_dir is not None else args.output_dir
            rede_outf = os.path.join(rede_dir, fname + '.png')
            rede_pil.save(rede_outf)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
