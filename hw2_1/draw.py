import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
def draw(G_losses,D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# def animation():
#     fig = plt.figure(figsize=(8, 8))
#     plt.axis("off")
#     ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
#     ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
#     HTML(ani.to_jshtml())
def real_fake(dataloader,img_list):
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[:32].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()