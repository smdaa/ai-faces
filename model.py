import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

LATENT_DIM = 128

# hyperparameters
lr = 1e-5
weight_decay = 1e-3

class Autoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.convnet1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128,
                      kernel_size=3, stride=2, padding=3),
            nn.ReLU()
        )

        self.convnet2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=2, padding=3),
            nn.ReLU()
        )

        self.convnet3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=3),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=172800, out_features=LATENT_DIM),
            nn.ReLU()
        )


        self.unbottleneck = nn.Sequential(
            nn.Linear(in_features=LATENT_DIM, out_features=172800),
            nn.ReLU()
        )

        self.deconvnet1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=2, padding=3, output_padding=(1, 1)),
            nn.ReLU()
        )

        self.deconvnet2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=2, padding=3, output_padding=(1, 0)),
            nn.ReLU()
        )

        self.deconvnet3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1,
                               kernel_size=3, stride=2, padding=3, output_padding=(0, 1)),
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
        #[bash_size, 128, 88, 95]
        out = self.convnet2(out)
        #[bash_size, 128, 46, 50]
        out = self.convnet3(out)
        #[bash_size, 256, 25, 27]
        out = self.bottleneck(out)
        #[bash_size, 64]
        return out

    def decode(self, x):

        out = self.unbottleneck(x)

        out = out.reshape(out.shape[0], 256, 25, 27)
        #[bash_size, 256, 25, 27]
        out = self.deconvnet1(out)
        #[bash_size, 128, 46, 50]
        out = self.deconvnet2(out)
        #[bash_size, 128, 88, 95]
        out = self.deconvnet3(out)
        #[bash_size, 128, 171, 186]

        return out

    def _get_reconstruction_loss(self, batch):
        x = batch[0].float()
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
