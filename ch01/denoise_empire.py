from PIL import Image
import pylab as pl
import numpy as np
from common import imtools

im = np.array(Image.open('data/empire.jpg').convert('L'))
U,T = imtools.denoise(im, im)

pl.figure('De-Noised Empire State Building')
pl.gray()
pl.imshow(U)
pl.axis('equal')
pl.axis('off')
pl.show()