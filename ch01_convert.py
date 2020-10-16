from PIL import Image
import os

pil_im = Image.open('empire.jpg').convert('L')
pil_im.save('empire_grayscale.jpg')
