from tkinter import *
import numpy as np
from model import *
from PIL import Image, ImageTk
import cv2

input_w = 186
input_h = 171
scale = 3
slider_from_value = -.1
slider_to_value = .1
num_params = 64
slides_perline = 16

device = torch.device('cuda')

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


class Window():

    def __init__(self, master):

        self.display1 = Canvas(master, width=input_w * scale,
                               height=input_h * scale, relief=RAISED)
        self.display1.grid(row=0, column=0)
        array = np.ones((input_h * scale, input_w * scale)) * 150
        img = ImageTk.PhotoImage(image=Image.fromarray(array))
        self.image_container = self.display1.create_image(
            0, 0, anchor='nw', image=img)

        self.ws = []
        self.ws_values = []

        for i in range(num_params):
            current_value = DoubleVar()
            self.ws_values.append(current_value)

        for i in range(num_params):
            slider = Scale(master, from_=slider_from_value, to=slider_to_value, orient=VERTICAL,
                           variable=self.ws_values[i], command=self.updateCanvas, resolution=0.01, length=100)
            self.ws.append(slider)

        for i in range(num_params):
            self.ws[i].grid(column=1 + i % slides_perline, row=1 +
                            i // slides_perline, sticky='we')

    def updateCanvas(self, ws_values):
        face_image = self.compute_image(self.ws_values)
        face_image = cv2.resize(face_image, dsize=(
            input_h * scale, input_w * scale), interpolation=cv2.INTER_CUBIC)
        img = ImageTk.PhotoImage(image=Image.fromarray(face_image))
        self.display1.imgref = img
        self.display1.itemconfig(self.image_container, image=img)

    def compute_image(self, ws_values):
        temp = np.array([x.get() for x in ws_values])
        x = means + np.dot(v, (temp * e).T).T
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float().to(device)
        face = model.decode(x).cpu().detach().numpy()

        return 255 * face[0, 0, :, :]


master = Tk()
w = Window(master)
master.mainloop()
