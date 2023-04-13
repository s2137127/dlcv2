


#https://github.com/tcapelle/Diffusion-Models-pytorch
import argparse
import random
from types import SimpleNamespace
from model import *
import torch

config = SimpleNamespace(
    run_name="DDPM_conditional",
    epochs=20,
    noise_steps=1000,
    seed=42,
    batch_size=32,
    img_size=24,
    num_classes=10,
    train_folder="train",
    val_folder="test",
    device="cuda",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=10,
    lr=3e-4)



def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    manualSeed = 666  # 666
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    diffuser.load()
    diffuser.fit(config)
