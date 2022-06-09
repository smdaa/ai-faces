import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *

# hyperparameters
batch_size = 128
epochs = 2000


def recontruct_image(x_train, x_test, model):
    index_train = random.sample(range(1, x_train.shape[0]), 3)
    index_test = random.sample(range(1, x_test.shape[0]), 3)

    x_train_s = x_train[index_train, :, :].float()
    x_test_s = x_test[index_test, :, :].float()

    x_train_s_r = model(x_train_s)
    x_test_s_r = model(x_test_s)

    x_train_s = x_train_s.cpu().detach().numpy()
    x_test_s = x_test_s.cpu().detach().numpy()

    x_train_s_r = x_train_s_r.cpu().detach().numpy()
    x_test_s_r = x_test_s_r.cpu().detach().numpy()

    fig, axs = plt.subplots(3, 4)

    for i in range(x_train_s.shape[0]):
        axs[i, 0].imshow(x_train_s[i, 0, :, :], cmap='gray')
        axs[i, 0].set_yticklabels([])
        axs[i, 0].set_xticklabels([])
        axs[i, 0].set_title('train sample' + str(i))

        axs[i, 1].imshow(x_train_s_r[i, 0, :, :], cmap='gray')
        axs[i, 1].set_yticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 1].set_title('train sample reconst' + str(i))

    for i in range(x_test_s.shape[0]):
        axs[i, 2].imshow(x_test_s[i, 0, :, :], cmap='gray')
        axs[i, 2].set_yticklabels([])
        axs[i, 2].set_xticklabels([])
        axs[i, 2].set_title('test sample' + str(i))

        axs[i, 3].imshow(x_test_s_r[i, 0, :, :], cmap='gray')
        axs[i, 3].set_yticklabels([])
        axs[i, 3].set_xticklabels([])
        axs[i, 3].set_title('test sample reconst' + str(i))

    plt.show()


# load data
x = torch.load('x.pt')

x = shuffle(x)
# train test split
x_train, x_test = train_test_split(x, test_size=0.2)

train_data = torch.utils.data.TensorDataset(x_train)
test_data = torch.utils.data.TensorDataset(x_test)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# load model
model = Autoencoder()
logger = TensorBoardLogger('cae_logs', name="cae")
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, logger=logger, callbacks=[
                     EarlyStopping(monitor="valid_loss", mode="min", patience=50)])

# train model
trainer.fit(model, train_loader, test_loader)

# save trained model
torch.save(model.state_dict(), './model.pt')

# visualise reconstructed images
recontruct_image(x_train, x_test, model)
