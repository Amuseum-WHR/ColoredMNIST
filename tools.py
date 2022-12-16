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

def change_color(t):
    ''' 
    t is a tensor with shape of (B,2,28,28),
    '''
    change_t = torch.zeros_like(t)
    change_t[:,0,:,:] = t[:,1,:,:]
    change_t[:,1,:,:] = t[:,0,:,:]
    return change_t

def judge_color(t):
    ''' 
    input:  t is a tensor with shape of (B,2,28,28)
    return: a vector (B, 1) value = 0 if green else 0
    '''
    return torch.round(torch.max(torch.max(t[:,0,:,:], dim=1)[0], dim=1)[0])