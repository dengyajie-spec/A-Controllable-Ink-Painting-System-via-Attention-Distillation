### The following section is from :-
### Zhou, Y., Gao, X., Chen, Z., & Huang, H. (2025, June). Attention distillation: A unified approach to visual characteristics transfer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 18270-18280). IEEE.
### https://github.com/xugao97/AttentionDistillation/blob/main/train_vae.py
###This implementation is not a direct verbatim copy. Several system‑specific adaptations are carried out: the code architecture is refactored for embedding within the web backend pipeline, additional GPU memory management is implemented, logging output is optimised, and the logic for saving intermediate images is eliminated.
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from diffusers import AutoencoderKL
from utils import load_image

class VAETrainer:
    def __init__(self, base_model_path="./stable-diffusion-v1-5", lr=1e-4, epochs=75, device="cuda"):
        self.base_model_path = base_model_path
        self.lr = lr
        self.epochs = epochs
        self.device = device

    def train(self, style_image_path, out_dir="./trained_vae_notebook"):
        """
        Fine‑tune the VAE Decoder and release related resources after training completes.
        """
        os.makedirs(out_dir, exist_ok=True)
        style_name = os.path.basename(style_image_path).split('.')[0]
        save_path = os.path.join(out_dir, f"trained_vae_{style_name}")

        print(f"[VAE Trainer] Loading standard VAE: {self.base_model_path}...")
        vae = AutoencoderKL.from_pretrained(
            self.base_model_path, 
            subfolder="vae",
            local_files_only=True
        ).to(self.device, dtype=torch.float32)
        
        vae.requires_grad_(False)

        # Load and normalize image
        image = load_image(style_image_path, size=(512, 512)).to(self.device, dtype=torch.float32)
        image_normalized = image * 2 - 1 

        with torch.no_grad():
            latents = vae.encode(image_normalized)["latent_dist"].mean

       # Enable gradient for decoder
        for param in vae.decoder.parameters():
            param.requires_grad = True
        vae.decoder.train()

        loss_fn = nn.L1Loss()
        optimizer = Adam(vae.decoder.parameters(), lr=self.lr)

        print(f"[VAE Trainer] Start VAE fine‑tuning for total {self.epochs} epochs...")
        for epoch in range(self.epochs):
            reconstructed = vae.decode(latents, return_dict=False)[0]
            loss = loss_fn(reconstructed, image_normalized)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 15 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

        # Save model and exit
        vae.save_pretrained(save_path)
        print(f"[VAE Trainer] Training finished, model saved to: {save_path}")
        
        # Explicitly delete variables and clear GPU cache to avoid OOM in subsequent generation stages
        del vae, optimizer, latents, image_normalized
        torch.cuda.empty_cache()
        
        return save_path
### end of Citation
