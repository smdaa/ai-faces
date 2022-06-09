from model import *
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")

# load trained model
model = Autoencoder()
model.load_state_dict(torch.load('./model.pt'))
model.to(device)

# load data
batch_size = 1024
x = torch.load('x.pt')
data = torch.utils.data.TensorDataset(x)
data_loader = torch.utils.data.DataLoader(
    dataset=data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# encode data
x_enc = np.empty((x.shape[0], LATENT_DIM))
i = 0
for data in data_loader:
    x_data = data[0].float()
    x_data = x_data.to(device)
    temp = model.encode(x_data)
    temp = temp.cpu().detach().numpy()
    x_enc[i:i+temp.shape[0], :] = temp 
    i = i + temp.shape[0]

# pca subspace base
x_mean = np.mean(x_enc, axis=0)
x_stds = np.std(x_enc, axis=0)
x_cov = np.cov((x_enc - x_mean).T)
e, v = np.linalg.eig(x_cov)

sort_inds = np.argsort(-e)
e = e[sort_inds]
v = v[:,sort_inds]

np.save('means.npy', x_mean)
np.save('stds.npy', x_stds)
np.save('e.npy', e)
np.save('v.npy', v)
