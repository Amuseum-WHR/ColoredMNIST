import torch
import numpy as np
from PIL import Image

def expand_to_3channels(t):
    zero = torch.zeros((1,28,28))
    t = torch.cat((t, zero))
    return t

def show_tensor(t):
    img = expand_to_3channels(t)
    nimg = img.numpy().transpose(1,2,0)
    img = nimg * 255 
    img = Image.fromarray(np.uint8(img)) # eg1
    img.show()