import numpy as np
from model import *
import tkinter as tk
from tkinter import ttk

device = torch.device('cpu')

input_w = 186
input_h = 171
scale = 3

# load autoencoder
model = Autoencoder()
model.load_state_dict(torch.load('./model.pt'))
model.to(device)

# load pca subspace
means = np.load('means.npy')
stds = np.load('stds.npy')
e = np.sqrt(np.load('e.npy'))
e = np.load('e.npy')
v = np.load('v.npy')

def _photo_image(image):
    height, width = image.shape
    data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


# manage window
root = tk.Tk()
root.geometry('1000x800')
root.title('ai-faces')

array = np.ones((input_h * scale, input_w * scale)) * 150
img = _photo_image(array)

canvas = tk.Canvas(root, width=input_w * scale, height=input_h * scale)
canvas.pack()
canvas.create_image(30, 10, anchor='nw', image=img)

root.mainloop()
