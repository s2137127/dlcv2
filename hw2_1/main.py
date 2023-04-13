#python -m pytorch_fid ./face/ ../hw2_data/hw2_data/face/val/
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from model import *
import time
from data import *
from draw import *
import random
import torch
import torchvision.utils as vutils
config = {
    'epochs': 88,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'LEARNING_RATE': 0.00003,
    'timestr':time.strftime("%d%H%M"),
    'b1':0.5,
    'b2':0.999

}

manualSeed = 477
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
nz = 100
fixed_noise = torch.randn(64, nz, 1, 1, device=config['DEVICE'])
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == '__main__':
    adversarial_loss = torch.nn.BCELoss()
    print(config['timestr'])
    print("cuda:",config['DEVICE'])
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    generator.to(config['DEVICE'])
    discriminator.to(config['DEVICE'])
    adversarial_loss.to(config['DEVICE'])

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    generator.load_state_dict(torch.load('./gen151827.pth'))
    discriminator.load_state_dict(torch.load('./dis151827.pth'))
    # # Configure data loader
    dataloader = get_dataloader()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['LEARNING_RATE']/1.5, betas=(config['b1'], config['b2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['LEARNING_RATE']*2, betas=(config['b1'], config['b2']))
    # optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=config['LEARNING_RATE']*1000,momentum=0.9)

    # ----------
    #  Training
    # ----------

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    print("Generater:",generator)
    print("########################################")
    print("Discrminator",discriminator)
    # For each epoch
    for epoch in range(config['epochs']):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 1):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()

            # Format batch
            real_cpu = data.to(config['DEVICE'])
            b_size = real_cpu.shape[0]
            r = 1
            # while r<0.7 or r>1.2:
            #     r = torch.rand(1).item()
            label = torch.full((b_size,), r, dtype=torch.float, device=config['DEVICE'])
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = adversarial_loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=config['DEVICE'])
            # Generate fake image batch with G
            fake = generator(noise)
            r = 0
            # while r < 0 or r > 0.3:
            #     r = torch.rand(1).item()
            label.fill_(r)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = adversarial_loss(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            r = 1
            # while r < 0.7 or r > 1.2:
            #     r = torch.rand(1).item()
            label.fill_(r)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = adversarial_loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config['epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config['epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                # real_fake(dataloader, img_list)
            iters += 10
            # if epoch in[15,30,45]:
            #     torch.save(generator.state_dict(), './gen%s_ep%s.pth' % (config['timestr'],epoch))
    torch.save(generator.state_dict(), './gen%s.pth' % config['timestr'])
    torch.save(discriminator.state_dict(), './dis%s.pth' % config['timestr'])

    draw(G_losses,D_losses)
    real_fake(dataloader,img_list)
    vutils.save_image(img_list[-1], "./model_B.png", nrow=8, normalize=True)
