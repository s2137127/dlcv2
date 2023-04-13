
from types import SimpleNamespace
from model import *
import torch
config = SimpleNamespace(
    noise_steps=1000,
    img_size=16,
    num_classes=10,
    device="cuda",
    slice_size=1,
    log_every_epoch=10,
    num_workers=10,
    lr=3e-4)
if __name__ == '__main__':
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    diffuser.load()
    # diffuser.log_images()#100*100img
    ##########################################################
    #######################################################
    img_step = diffuser.sample_show_step().float()#200step per img of zero
    vutils.save_image(img_step, normalize=True, fp="./zero_sample2.png")
