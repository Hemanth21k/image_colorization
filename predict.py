# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time
from model import ColorizationNet
import PIL
# from IPython.display import Image, 
import tkinter as tk
from tkinter import filedialog
import cv2

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

weights = "model-final-model-losses-0.003.pth"

device = torch.device('cpu')

model = ColorizationNet()
model.load_state_dict(torch.load(weights,map_location=device))

img_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
img = PIL.Image.open(file_path)
pim = torch.from_numpy(rgb2gray(np.asarray(img_transforms(img)))).unsqueeze(0).float()
model.eval()
pm = model(pim.unsqueeze(0))
mm = torch.cat((pim, pm.squeeze(0).cpu()), 0).detach().numpy()
mm = mm.transpose((1, 2, 0))  # rescale for matplotlib
mm[:, :, 0:1] = mm[:, :, 0:1] * 100
mm[:, :, 1:3] = mm[:, :, 1:3] * 255 - 128   
color_image = lab2rgb(mm.astype(np.float64))
# cv2.imshow("Original",color_image)

plt.imsave(arr=color_image, fname='output.png')