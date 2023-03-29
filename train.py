from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

from networks import Generator, Discriminator
from utils import GAN, Trainer

train = MNIST("./dataset/train", train=True, download=False,
              transform=transforms.Compose([transforms.ToTensor()]))
test = MNIST("./dataset/test", train=False, download=False,
             transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train, batch_size=16, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=True)

generator_net = Generator(7*7, 10).cuda()
discriminator_net = Discriminator((28, 28)).cuda()
gan = GAN(generator_net, discriminator_net, generator_lr=0.00001, discriminator_lr=0.00001)

trainer = Trainer(gan, train_loader, 250, 3, 10)
trainer.train()