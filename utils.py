import matplotlib.pyplot as plt
import torch

from torch.nn.functional import binary_cross_entropy, mse_loss
from torch.nn import BCELoss
from torch.optim import Adam

from torchvision import transforms
from torchmetrics import Accuracy


class GAN:
    def __init__(self, generator_net, discriminator_net, generator_lr=0.001, discriminator_lr=0.001):
        self.generator = generator_net
        self.latent_dim = generator_net.latent_input_size
        self.discriminator = discriminator_net
        self.generator_optim = Adam(self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        self.criterion = BCELoss()

    def latent_vector_generator(self, size, label=-1):
        latent_input = torch.randn(self.latent_dim * size)
        latent_input = latent_input.reshape(size, self.latent_dim)

        if label == -1:
            cat_labels = torch.randint(0, 10, (size, 1))
        else:
            cat_labels = torch.tensor([label])

        return [latent_input.detach(), cat_labels.detach()]

    def train_generator(self, size):
        self.generator_optim.zero_grad()
        latent_input, cat_labels = self.latent_vector_generator(size)
        generator_out = self.generator(latent_input.detach().cuda(), cat_labels.detach().cuda())
        disc_out = self.discriminator(generator_out, cat_labels.cuda())
        y_true = torch.ones((size, 1), dtype=torch.float).cuda()
        generator_loss = self.criterion(disc_out, y_true.detach())
        generator_loss.backward()
        self.generator_optim.step()

    def generate_fake_batch(self, size):
        latent_inputs, cat_labels = self.latent_vector_generator(size)
        fake_images = self.generator(latent_inputs.cuda(), cat_labels.cuda())
        y_trues = torch.zeros((size, 1), dtype=torch.float).detach().cuda()
        return fake_images.detach(), cat_labels.detach().cuda(), y_trues

    def discriminator_train(self, batch_x, batch_y):
        self.discriminator_optim.zero_grad()
        fake_images, cat_labels, y_true_zeros = self.generate_fake_batch(len(batch_x))
        real_pred = self.discriminator(batch_x.detach(), batch_y.unsqueeze(dim=1).detach())
        real_y = torch.ones((len(batch_x), 1), dtype=torch.float).cuda()
        fake_pred = self.discriminator(fake_images, cat_labels)
        fake_y = y_true_zeros
        real_loss = self.criterion(real_pred, real_y.detach())
        fake_loss = self.criterion(fake_pred, fake_y.detach())
        real_loss.backward()
        fake_loss.backward()
        # total_loss = (real_loss + fake_loss) * 0.5
        # assert torch.isnan(total_loss).item() == False, "Something goes wrong! One or more loss values contains 'nan' value. "
        # total_loss.backward()
        self.discriminator_optim.step()

    def test(self, test_batch_size):
        latent_vector, cat_labels = self.latent_vector_generator(test_batch_size)

        fake_img = self.generator(latent_vector.cuda(), cat_labels.cuda()).detach()
        disc_out = self.discriminator(fake_img, cat_labels.cuda()).detach()
        ones = torch.ones((test_batch_size, 1), dtype=torch.float).cuda()
        zeros = torch.zeros((test_batch_size, 1), dtype=torch.float).cuda()
        disc_loss = binary_cross_entropy(disc_out, zeros)
        generative_loss = binary_cross_entropy(disc_out, ones)
        self.discriminator_optim.zero_grad()
        self.generator_optim.zero_grad()
        return generative_loss, disc_loss, fake_img[0], cat_labels[0]


class Trainer:
    def __init__(self, gan, train_dataloader, total_epochs, discriminator_epochs, test_batch_size):
        self.dataloader = train_dataloader
        self.batch_iter = len(train_dataloader.dataset) // train_dataloader.batch_size
        self.total_epochs = total_epochs
        self.discriminator_epochs = discriminator_epochs
        self.gan = gan
        self.test_batch_size = test_batch_size

    @staticmethod
    def sample_one_test(fake_img, label, epoch_num):
        fake_img = (fake_img+1) * 0.5
        transform = transforms.ToPILImage()
        img = transform(fake_img)
        plt.title(f"{label.item()}")
        plt.imshow(img, cmap="gray")
        plt.savefig(f"./figs/{epoch_num}_{label.item()}.png")
        plt.show()

    def __train_disc(self):
        for _, batch in enumerate(self.dataloader):
            for param in self.gan.discriminator.parameters(): param.requires_grad = True
            x = batch[0].cuda()
            y = batch[1].cuda()
            self.gan.discriminator_train(x, y)

    def __train_gen(self):
        for param in self.gan.discriminator.parameters():
            param.requires_grad = False
        for _, batch in enumerate(self.dataloader):
            self.gan.train_generator(self.dataloader.batch_size * 2)

    def train(self):
        flag = "train_gen"
        for epoch in range(605, self.total_epochs):
            if epoch < self.discriminator_epochs:
                self.__train_disc()
                continue

            if epoch % 4 == 0 or flag == "train_disc":
                self.__train_disc()
                flag = "train_gen"
            else:
                self.__train_gen()

            generative_loss, disc_loss, fake_img, cat_label = self.gan.test(self.test_batch_size)
            print(f"Epoch: {epoch} --- Generative loss: {generative_loss} --- Disc loss: {disc_loss}")
            if disc_loss > 1.0 or disc_loss > generative_loss:
                flag = "train_disc"

            if epoch % 2 == 0:
                Trainer.sample_one_test(fake_img.detach(), cat_label.detach(), epoch_num=epoch)
                torch.save(self.gan.generator.state_dict(), f"./models/generator/{epoch}.pth")
                torch.save(self.gan.discriminator.state_dict(), f"./models/discriminator/{epoch}.pth")


class Tester:
    def __init__(self, gan, ):  #test_dataloader
        # self.dataloader = test_dataloader
        self.gan = gan
        self.discriminator = gan.discriminator
        self.generator = gan.generator

    # def classifier(self):
    #     for batch_idx, batch in enumerate(self.dataloader):
    #         x = batch[0].cuda()
    #         y = batch[1].cuda()
    #         y_pred = self.discriminator(x, y).detach().long()
    #         y_true = torch.ones((len(batch[0]), 1)).detach().cuda()
    #         ac = Accuracy().cuda()
    #         print(ac(y_pred.long(), y_true.long()).item())

    def inference(self, input_num):
        latent_vector, cat_label = self.gan.latent_vector_generator(size=1, label=input_num)
        out = self.generator(latent_vector.cuda(), cat_label.cuda())[0].detach()
        out = (out+1)*0.5   # rescaling to -1,1 to 0,1
        transform = transforms.ToPILImage()
        img = transform(out)
        return img
