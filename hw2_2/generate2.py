
from types import SimpleNamespace

import sys
from os.path import isdir, dirname
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.utils.data
import copy
from unet import *
import os
from torchvision import transforms
import torchvision.utils as vutils
config = SimpleNamespace(
    noise_steps=1000,
    img_size=16,
    num_classes=10,
    device="cuda",
    slice_size=1,
    num_workers=10,
    lr=3e-4)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=24, num_classes=10, c_in=3, c_out=3,
                 device="cuda"):
        self.train_dataloader = None
        self.val_dataloader = None
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, n, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.inference_mode():
            x = np.random.randn(n, self.c_in, self.img_size, self.img_size)
            for i in reversed(range(0, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                x_ten = torch.from_numpy(x).to(self.device).float()
                # predicted_noise = model(x_ten, t, labels).cpu().numpy()
                predicted_noise = model(x_ten, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x_ten, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale).cpu().numpy()
                alpha = self.alpha[t][:, None, None, None].cpu().numpy()
                alpha_hat = self.alpha_hat[t][:, None, None, None].cpu().numpy()
                beta = self.beta[t][:, None, None, None].cpu().numpy()
                # t2 = time.time()

                if i > 1:
                    noise = np.random.randn(*x.shape)
                else:
                    noise = np.zeros_like(x)

                x = 1 / np.sqrt(alpha) * (
                        x - ((1 - alpha) / (np.sqrt(1 - alpha_hat))) * predicted_noise) + np.sqrt(
                    beta) * noise
                # x = torch.from_numpy(x).to(self.device)
        x = torch.from_numpy(x).to(self.device).float()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    @torch.inference_mode()
    def log_one_image(self):
        output = sys.argv[1]
        if not isdir(dirname(output)):
            os.mkdir(dirname(output))
        num =1
        for i in range(10):
            labels = torch.tensor(i, device=self.device).long().expand(100)

            sampled_images = self.sample(use_ema=False, n=len(labels), labels=labels)
            # t0 = time.time()
            for j in range(sampled_images.shape[0]):
                x = transforms.Resize((28,28))(sampled_images[j])
                vutils.save_image(vutils.make_grid(x.float(), normalize=True), os.path.join(output,'%d_%03d.png' %(i,num)))
                num +=1
                if num>100:
                    num = 1
                # t1 = time.time()
                # print("time:",t1-t0)
    def load(self):
        self.model.load_state_dict(torch.load('./DDPM_conditional_model_19.pth'))
        # self.ema_model.load_state_dict(torch.load('./DDPM_conditional_ema2.pth'))


if __name__ == '__main__':
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    diffuser.load()
    diffuser.log_one_image()