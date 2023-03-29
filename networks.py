import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_input_size, class_size):
        super(Generator, self).__init__()
        self.latent_input_size = latent_input_size
        self.class_size = class_size
        self.label_embed = nn.Embedding(self.class_size, 50)
        self.label_linear = nn.Linear(50, 7*7)

        self.latent_linear = nn.Sequential(
            nn.Linear(self.latent_input_size, 7 * 7 * 128),
            nn.ReLU(),
        )
        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(129, 128, 4, stride=2, padding=1), # out shape =   14*14*128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), # out shape =   28*28*128
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=7, padding="same"), # out shape = 28*28*1
            nn.Tanh(),
        )

    def forward(self, latent, label):
        embedded = self.label_embed(label)
        label_out = self.label_linear(embedded)
        label_out = label_out.view(-1, 1, 7, 7)

        latent_linear_out = self.latent_linear(latent)
        latent_linear_out = latent_linear_out.view(-1, 128, 7, 7)
        concat = torch.concat([latent_linear_out, label_out], dim=1)
        conv_out = self.conv_net(concat)
        return conv_out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape_x = input_shape[0]
        self.input_shape_y = input_shape[1]
        self.label_embed = nn.Embedding(10, 50)
        self.label_linear = nn.Linear(50, 28*28)
        self.conv_net = nn.Sequential(                 # 28*28*2
            nn.Conv2d(2, 32, 3, padding=1),  # 28*28*64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),              # 14*14*64
            nn.Conv2d(32, 64, 3, padding=1),  # 14*14*128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 7*7*128
            nn.Conv2d(64, 128, 4, padding=1),  # 7*7*128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 3*3*128
        )

        self.linear = nn.Sequential(
            nn.Linear(3*3*128, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        label_embed = self.label_embed(label)
        label_out = self.label_linear(label_embed)
        label_out = label_out.view(-1, 1, 28, 28)
        concat = torch.concat([img, label_out], dim=1)
        conv_out = self.conv_net(concat)
        if conv_out.shape == torch.Size([32, 28, 28]):
            flatten = torch.flatten(conv_out, start_dim=0)
        else:
            flatten = torch.flatten(conv_out, start_dim=1)
        linear_out = self.linear(flatten)

        return linear_out
