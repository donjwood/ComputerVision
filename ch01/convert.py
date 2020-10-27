from PIL import Image
import os

pil_im = Image.open('images/empire.jpg').convert('L')
pil_im.save('images/empire_grayscale.jpg')
