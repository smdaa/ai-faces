import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

LATENT_DIM = 64

# hyperparameters
lr = 1e-5
#weight_decay = 1e-3
momentum = 0.9


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.convnet1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, stride=3, padding=3),
            nn.ReLU()
        )

        self.convnet2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=3, padding=3),
            nn.ReLU()
        )

        self.convnet3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, stride=3, padding=3),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=LATENT_DIM),
            nn.ReLU()
        )

        self.unbottleneck = nn.Sequential(
            nn.Linear(in_features=LATENT_DIM, out_features=8192),
            nn.ReLU()
        )

        self.deconvnet1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=5, stride=3, padding=3, output_padding=(0, 2)),
            nn.ReLU()
        )

        self.deconvnet2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=5, stride=3, padding=3, output_padding=(2, 1)),
            nn.ReLU()
        )

        self.deconvnet3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1,
                               kernel_size=5, stride=3, padding=3, output_padding=(1, 1)),
            nn.Sigmoid()
        )
        self.example_input_array = torch.zeros(2, 1, 171, 186)

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out

    def encode(self, x):
        #[bash_size, 1, 171, 186]
        out = self.convnet1(x)
        #[bash_size, 32, 58, 63]
        out = self.convnet2(out)
        #[bash_size, 64, 20, 22]
        out = self.convnet3(out)
        #[bash_size, 128, 8, 8]
        out = self.bottleneck(out)
        #[bash_size, 128]
        return out

    def decode(self, x):
        out = self.unbottleneck(x)
        out = out.reshape(out.shape[0], 128, 8, 8)
        #[bash_size, 128, 8, 8]
        out = self.deconvnet1(out)
        #[bash_size, 64, 20, 22]
        out = self.deconvnet2(out)
        #[bash_size, 32, 58, 63]
        out = self.deconvnet3(out)
        #[bash_size, 1, 171, 186]

        return out

    def _get_reconstruction_loss(self, batch):
        x = batch[0].float()
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
