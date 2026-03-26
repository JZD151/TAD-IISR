import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_lora_fwd, my_vae_encoder_fwd, my_vae_decoder_fwd
from basicsr.archs.arch_util import default_init_weights
from my_utils.vaehook import VAEHook, perfcount

class TAD_IISR(torch.nn.Module):
    def __init__(
        self,
        sd_path=None,
        pretrained_path=None,
        lora_rank_unet=32,
        lora_rank_vae=16,
        block_embedding_dim=64,
        args=None,
        deg_map_in_chans: int = 64,
    ):
        super().__init__()
        self.args = args
        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)
        self.guidance_scale = 1.07

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        # Attach pix2pix_turbo-style skip connections between VAE encoder and decoder.
        # Monkey-patch encoder/decoder forward to expose encoder features and inject them in decoder.
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # Add the skip connection convs (will be moved to cuda with vae.to(...))
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1.0

        # NOTE: keep this consistent with training-time model.
        target_modules_vae = r"^(encoder|decoder)\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|skip_conv_\d+|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        # Support degradation condition as either:
        # - vector: (B, 64)
        # - map: (B, deg_map_in_chans, H, W) from de_enc_net
        self.deg_map_in_chans = deg_map_in_chans
        self.deg_map_to_vec = nn.Sequential(
            nn.Conv2d(self.deg_map_in_chans, num_embeddings, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 1, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 1, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp, self.deg_map_to_vec], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")

            # Rank sanity check (must match to avoid reshape crashes)
            if "rank_vae" in sd and int(sd["rank_vae"]) != int(lora_rank_vae):
                raise ValueError(
                    f"Checkpoint rank_vae={int(sd['rank_vae'])} != current lora_rank_vae={int(lora_rank_vae)}. "
                    "Please pass the same lora_rank_vae as used in the checkpoint."
                )
            if "rank_unet" in sd and int(sd["rank_unet"]) != int(lora_rank_unet):
                raise ValueError(
                    f"Checkpoint rank_unet={int(sd['rank_unet'])} != current lora_rank_unet={int(lora_rank_unet)}. "
                    "Please pass the same lora_rank_unet as used in the checkpoint."
                )

            # Informative warnings when target modules differ.
            if "vae_lora_target_modules" in sd and sd["vae_lora_target_modules"] != target_modules_vae:
                warnings.warn(
                    "Checkpoint 'vae_lora_target_modules' differs from current target_modules_vae. "
                    "Loading will proceed; unmatched keys will be skipped."
                )
            if "unet_lora_target_modules" in sd and sd["unet_lora_target_modules"] != target_modules_unet:
                warnings.warn(
                    "Checkpoint 'unet_lora_target_modules' differs from current target_modules_unet. "
                    "Loading will proceed; unmatched keys will be skipped."
                )

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)

            def _safe_update_state_dict(dst_sd, src_sd, prefix: str):
                loaded = 0
                skipped = 0
                for k, v in src_sd.items():
                    if k in dst_sd and dst_sd[k].shape == v.shape:
                        dst_sd[k] = v
                        loaded += 1
                    else:
                        skipped += 1
                if skipped > 0:
                    warnings.warn(f"[{prefix}] loaded {loaded} keys, skipped {skipped} (missing or shape mismatch)")
                return dst_sd

            if "state_dict_unet" in sd:
                _sd_unet = _safe_update_state_dict(unet.state_dict(), sd["state_dict_unet"], prefix="unet")
                unet.load_state_dict(_sd_unet, strict=False)
            else:
                warnings.warn("Checkpoint missing 'state_dict_unet'; UNet LoRA/conv_in will stay initialized")

            if "state_dict_vae" in sd:
                _sd_vae = _safe_update_state_dict(vae.state_dict(), sd["state_dict_vae"], prefix="vae")
                vae.load_state_dict(_sd_vae, strict=False)
            else:
                warnings.warn("Checkpoint missing 'state_dict_vae'; VAE LoRA/skip_conv will stay initialized")

            if "state_dict_vae_de_mlp" in sd:
                self.vae_de_mlp.load_state_dict(sd["state_dict_vae_de_mlp"], strict=False)
            if "state_dict_unet_de_mlp" in sd:
                self.unet_de_mlp.load_state_dict(sd["state_dict_unet_de_mlp"], strict=False)
            if "state_dict_vae_block_mlp" in sd:
                self.vae_block_mlp.load_state_dict(sd["state_dict_vae_block_mlp"], strict=False)
            if "state_dict_unet_block_mlp" in sd:
                self.unet_block_mlp.load_state_dict(sd["state_dict_unet_block_mlp"], strict=False)
            if "state_dict_vae_fuse_mlp" in sd:
                self.vae_fuse_mlp.load_state_dict(sd["state_dict_vae_fuse_mlp"], strict=False)
            if "state_dict_unet_fuse_mlp" in sd:
                self.unet_fuse_mlp.load_state_dict(sd["state_dict_unet_fuse_mlp"], strict=False)
            if "state_dict_deg_map_to_vec" in sd:
                self.deg_map_to_vec.load_state_dict(sd["state_dict_deg_map_to_vec"], strict=False)

            if "state_embeddings" in sd:
                emb = sd["state_embeddings"]
                if isinstance(emb, dict) and "state_dict_vae_block" in emb:
                    self.vae_block_embeddings.load_state_dict(emb["state_dict_vae_block"], strict=False)
                if isinstance(emb, dict) and "state_dict_unet_block" in emb:
                    self.unet_block_embeddings.load_state_dict(emb["state_dict_unet_block"], strict=False)

            if "w" in sd and isinstance(sd["w"], torch.Tensor) and self.W.shape == sd["w"].shape:
                self.W.data.copy_(sd["w"].to(self.W.data.dtype))
            elif "W" in sd and isinstance(sd["W"], torch.Tensor) and self.W.shape == sd["W"].shape:
                self.W.data.copy_(sd["W"].to(self.W.data.dtype))
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()
        self.deg_map_to_vec.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)
        self.vae_de_mlp.requires_grad_(False)
        self.unet_de_mlp.requires_grad_(False)
        self.vae_block_mlp.requires_grad_(False)
        self.unet_block_mlp.requires_grad_(False)
        self.vae_fuse_mlp.requires_grad_(False)
        self.unet_fuse_mlp.requires_grad_(False)
        self.deg_map_to_vec.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        if hasattr(self.vae, "decoder") and hasattr(self.vae.decoder, "skip_conv_1"):
            self.vae.decoder.skip_conv_1.requires_grad_(False)
            self.vae.decoder.skip_conv_2.requires_grad_(False)
            self.vae.decoder.skip_conv_3.requires_grad_(False)
            self.vae.decoder.skip_conv_4.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp.train()
        self.unet_de_mlp.train()
        self.vae_block_mlp.train()
        self.unet_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.unet_fuse_mlp.train()
        self.deg_map_to_vec.train()

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        self.vae_block_embeddings.requires_grad_(True)
        self.unet_block_embeddings.requires_grad_(True)
        self.vae_de_mlp.requires_grad_(True)
        self.unet_de_mlp.requires_grad_(True)
        self.vae_block_mlp.requires_grad_(True)
        self.unet_block_mlp.requires_grad_(True)
        self.vae_fuse_mlp.requires_grad_(True)
        self.unet_fuse_mlp.requires_grad_(True)
        self.deg_map_to_vec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        if hasattr(self.vae, "decoder") and hasattr(self.vae.decoder, "skip_conv_1"):
            self.vae.decoder.skip_conv_1.requires_grad_(True)
            self.vae.decoder.skip_conv_2.requires_grad_(True)
            self.vae.decoder.skip_conv_3.requires_grad_(True)
            self.vae.decoder.skip_conv_4.requires_grad_(True)

    @perfcount
    @torch.no_grad()
    def forward(self, c_t, deg_score, pos_prompt, neg_prompt=None):
 
        if pos_prompt is not None:
            # encode the text prompt
            pos_caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            pos_caption_enc = self.text_encoder(pos_caption_tokens)[0]
        else:
            pos_caption_enc = self.text_encoder(prompt_tokens)[0]

        if neg_prompt is not None:
            # encode the text prompt
            neg_caption_tokens = self.tokenizer(neg_prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            neg_caption_enc = self.text_encoder(neg_caption_tokens)[0]
            
        if deg_score.dim() == 4:
            deg_proj = self.deg_map_to_vec(deg_score)
        else:
            deg_proj = deg_score

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")
                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
            if neg_prompt is not None:
                neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
                model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
            else:
                model_pred = pos_model_pred
        else:
            print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")

            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(c_t.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        pos_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                        if neg_prompt is not None:
                            neg_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
                            model_out = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
                        else:
                            model_out = pos_model_pred
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample

        # Provide encoder activations to decoder for skip connections
        if hasattr(self.vae, "encoder") and hasattr(self.vae.encoder, "current_down_blocks"):
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}
        sd["state_dict_deg_map_to_vec"] = {k: v for k, v in self.deg_map_to_vec.state_dict().items()}
        sd["w"] = self.W

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }

        torch.save(sd, outf)

    def _set_latent_tile(self,
        latent_tiled_size = 96,
        latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap
    
    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))

