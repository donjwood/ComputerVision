from PIL import Image
import os

pil_im = Image.open('data/empire.jpg').convert('L')
pil_im.save('data/empire_grayscale.jpg')
