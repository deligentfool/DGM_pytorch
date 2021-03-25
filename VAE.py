import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image


class variational_autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(variational_autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(128, self.latent_dim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(128, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim)
        )

    def encode(self, x):
        encoder = self.encoder(x)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


if __name__ == '__main__':
    img_size = 28
    img_channel = 1
    img_shape = [img_channel, img_size, img_size]
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    learning_rate = 0.0002
    max_epoch = 200
    store_freq = 400

    os.makedirs(name='./data/mnist', exist_ok=True)
    os.makedirs(name='./imgs/vae', exist_ok=True)
    dataloader = DataLoader(
        datasets.MNIST(
            root='./data/mnist',
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            ),
            download=False
        ),
        batch_size=batch_size,
        shuffle=True
    )
    vae = variational_autoencoder(img_size * img_size, latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(max_epoch):
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.view(imgs.size(0), -1).to(device)
            output_imgs, mu, logvar = vae.forward(imgs)
            mse_loss = (imgs - output_imgs).pow(2).mean()
            kld_loss = (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = -0.5 * kld_loss + mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[Epoch {}/{}] [Batch {}/{}] [loss: {:.7f}]'.format(epoch, max_epoch, i, len(dataloader), loss.item()))
            batches_done = i + epoch * len(dataloader)
            if batches_done % store_freq == 0:
                output_imgs = output_imgs.view(imgs.size(0), * img_shape)
                save_image(output_imgs[:25], './imgs/vae/{}.png'.format(batches_done), nrow=5, normalize=True)