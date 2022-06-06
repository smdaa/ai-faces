import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_image(fname):
    image = plt.imread(fname)
    image = np.sum(image, axis=2) / image.shape[2]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def load_images(data_path):
    n = 0
    x = np.empty((n, 171, 186))

    for filename in os.listdir(data_path):
        if filename == 'F' or filename == 'M':
            for fname in os.listdir(os.path.join(data_path, filename)):
                n = n + 1

    x = np.empty((n, 171, 186))

    i = 0
    for filename in os.listdir(data_path):
        if filename == 'F' or filename == 'M':
            for fname in os.listdir(os.path.join(data_path, filename)):
                image = load_image(os.path.join(
                    data_path, os.path.join(filename, fname)))
                x[i, :, :] = image
                i = i + 1

    return x


data_path = './data'
x = load_images(data_path)
x = np.expand_dims(x, axis=1)
x = torch.from_numpy(x)
torch.save(x, 'x.pt')
