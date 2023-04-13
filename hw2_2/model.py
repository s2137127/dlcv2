import sys
from os.path import isdir, dirname

from data import *
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.utils.data
import random
import copy
from unet import *
import os
from torchvision import transforms
import torchvision.utils as vutils
manualSeed = 666#666
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

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



    def sample_show_step(self, use_ema=False, n=1, labels=torch.tensor(0,device='cuda').long(), cfg_scale=3):
        model = self.ema_model if use_ema else self.model

        model.eval()
        with torch.inference_mode():
            tmp = None
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
                if i % 200 ==0 or i == self.noise_steps-1:
                    if i == self.noise_steps-1:
                        tmp = x
                    else:
                        tmp = torch.concat((tmp,x),dim=0)
        tmp = (tmp.clamp(-1, 1) + 1) / 2
        tmp = (tmp * 255).type(torch.uint8)
        return tmp

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train:
            self.model.train()
            dt = self.train_dataloader
        else:
            self.model.eval()
            dt = self.val_dataloader
        for i, (images, labels) in enumerate(tqdm(dt)):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.step_ema(self.ema_model, self.model)
                self.scheduler.step()
            if(i%5000) ==0 or i ==len(self.train_dataloader):
                print(f"MSE={loss.item():2.3f}")

        return avg_loss.mean().item()

    @torch.inference_mode()
    def log_images(self):
        all_img = None
        for ll in range(10):
            print(ll)

            labels = torch.tensor(ll,device=self.device).long().expand(10)
            sampled_images = self.sample(use_ema=False, n=len(labels), labels=labels)
            if ll == 0:
                all_img = sampled_images
            else:
                all_img = torch.concat((all_img,sampled_images),dim=0)
        vutils.save_image(all_img.float(),normalize=True,nrow=10, fp="./test_sample.png")

    def log_one_image(self):
        output = sys.argv[1]
        if not isdir(dirname(output)):
            os.mkdir(dirname(output))
        num =1
        # labels = torch.arange(10, device=self.device).long().expand(100, 10)
        # labels = torch.transpose(labels, 0, 1).reshape(-1)
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

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), './DDPM_conditional_model2.pth')
        torch.save(self.ema_model.state_dict(), './DDPM_conditional_ema2.pth')
        torch.save(self.optimizer.state_dict(), './DDPM_conditional_optim2.pth')


    def fit(self, args):
        print(self.model)
        device = args.device
        self.train_dataloader, self.val_dataloader = get_dataloader()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            print("epoch:",epoch)
            loss = self.one_epoch(train=True)
            print("train_loss:",loss)
            ## validation

            avg_loss = self.one_epoch(train=False)
            print("val_loss:", avg_loss)
            if epoch%10==0 or epoch == args.epochs-1:
                # save model
                self.save_model(run_name=args.run_name, epoch=epoch)

