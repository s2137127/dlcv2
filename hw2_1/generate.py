import torch
import torchvision.utils as vutils
import random
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os,sys
from os import mkdir
from os.path import isdir, dirname
from model import Generator
manualSeed = 666#666
#manualSeed = random.randint(1, 10000) 
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
fixed_noise = torch.randn(1000, 100, 1, 1, device='cuda')
if __name__ == '__main__':
    output = sys.argv[1]
    if not isdir(dirname(output)):
        mkdir(dirname(output))
    netG = Generator().to('cuda')
    name = 'gen152057.pth'
    print(name)
    netG.load_state_dict(torch.load(name))
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        for i in range(fake.size(0)):
            vutils.save_image(fake[i], os.path.join(output,"%s.png" %i), nrow=1, normalize=True)


