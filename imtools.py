import os
from PIL import Image
from numpy import *

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# Resize an image array using PIL
def imresize(im, sz):
    pil_im = Image.fromarray(uint8(im))

    return array(pil_im.resize(sz))