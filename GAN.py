import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image


class generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        def block(in_dim, out_dim, norm=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if norm:
                layers.append(nn.BatchNorm1d(out_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            * block(in_dim=self.input_dim, out_dim=128, norm=False),
            * block(in_dim=128, out_dim=256),
            * block(in_dim=256, out_dim=512),
            * block(in_dim=512, out_dim=1024),
            nn.Linear(1024, self.output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    img_size = 28
    img_channel = 1
    img_shape = [img_channel, img_size, img_size]
    batch_size = 64
    cuda = torch.cuda.is_available()
    noise_dim = 100
    learning_rate = 0.0002
    max_epoch = 200
    D_update_iter = 3
    store_freq = 400

    os.makedirs(name='./data/mnist', exist_ok=True)
    os.makedirs(name='./imgs/gan', exist_ok=True)
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
    G = generator(noise_dim, np.prod(img_shape))
    D = discriminator(np.prod(img_shape))
    loss_func = torch.nn.BCELoss()

    if cuda:
        G = G.cuda()
        D = D.cuda()
        loss_func = loss_func.cuda()

    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
    for epoch in range(max_epoch):
        for i, (imgs, _) in enumerate(dataloader):
            real_labels = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).cuda()
            fake_labels = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).cuda()

            real_imgs = imgs.view(imgs.size(0), -1).cuda()

            z = np.random.normal(0, 1, size=[imgs.size(0), noise_dim])
            z = torch.FloatTensor(z).cuda()
            fake_imgs = G.forward(z)

            D_loss_list = []
            for _ in range(D_update_iter):
                real_loss = loss_func(D.forward(real_imgs), real_labels)
                fake_loss = loss_func(D.forward(fake_imgs.detach()), fake_labels)

                D_optimizer.zero_grad()
                D_loss = (real_loss + fake_loss) / 2
                D_loss_list.append(D_loss.item())
                D_loss.backward()
                D_optimizer.step()

            z = np.random.normal(0, 1, size=[imgs.size(0), noise_dim])
            z = torch.FloatTensor(z).cuda()
            fake_imgs = G.forward(z)

            G_optimizer.zero_grad()
            G_loss = loss_func(D.forward(fake_imgs), real_labels)
            G_loss.backward()
            G_optimizer.step()

            print('[Epoch {}/{}] [Batch {}/{}] [D loss: {:.7f}] [G loss: {:.7f}]'.format(epoch, max_epoch, i, len(dataloader), np.mean(D_loss_list), G_loss.item()))
            batches_done = i + epoch * len(dataloader)
            if batches_done % store_freq == 0:
                fake_imgs = fake_imgs.view(imgs.size(0), * img_shape)
                save_image(fake_imgs[:25], './imgs/gan/{}.png'.format(batches_done), nrow=5, normalize=True)