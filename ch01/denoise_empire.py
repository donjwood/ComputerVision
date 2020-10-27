from PIL import Image
import pylab as pl
from common import imtools

im = pl.array(Image.open('images/empire.jpg').convert('L'))
U,T = imtools.denoise(im, im)

pl.figure('De-Noised Empire State Building')
pl.gray()
pl.imshow(U)
pl.axis('equal')
pl.axis('off')
pl.show()