import matplotlib.pyplot as plt
import os
os.chdir("/home/emredo/CGAN_MNIST")
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from networks import Generator, Discriminator
from utils import GAN, Tester

# train = MNIST("./dataset/train", train=True, download=False,
#               transform=transforms.Compose([transforms.ToTensor()]))
# test = MNIST("./dataset/test", train=False, download=False,
#              transform=transforms.Compose([transforms.ToTensor()]))
#
# train_loader = DataLoader(train, batch_size=16, shuffle=True)
# test_loader = DataLoader(test, batch_size=64, shuffle=True)

generator_net = Generator(7*7, 10).cuda()
discriminator_net = Discriminator((28, 28)).cuda()
discriminator_net.load_state_dict(torch.load("./models/discriminator/652.pth"))
generator_net.load_state_dict(torch.load("./models/generator/652.pth"))

gan = GAN(generator_net, discriminator_net)

tester = Tester(gan)
while True:
    inp = input("Please type a number between [0,9]:\t")
    if inp == "q":
        break
    try:
        inp = int(inp)
    except:
        print("Please type NUMBER.")
    img = tester.inference(inp)
    plt.title(f"{inp}")
    plt.imshow(img, cmap="gray")
    plt.show()