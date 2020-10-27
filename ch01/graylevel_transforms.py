from PIL import Image
import numpy as np
import pylab as pl

im = np.array(Image.open('images/empire.jpg').convert('L'))

im2 = 255 - im # invert image

im3 = (100.0/255) * im + 100 # clamp to interval 100..200

im4 = 255.0 * (im/255.0)**2 # squared

# Output min/max to console
print('Figure 1 min/max: ' + str(im.min()) + '/' + str(im.max()))
print('Figure 2 min/max: ' + str(im2.min()) + '/' + str(im2.max()))
print('Figure 3 min/max: ' + str(im3.min()) + '/' + str(im3.max()))
print('Figure 4 min/max: ' + str(im4.min()) + '/' + str(im4.max()))


pl.figure('Original (f(x)=x)')
pl.gray()
pl.imshow(im)
pl.figure('Inversion (f(x)=255-x)')
pl.gray()
pl.imshow(im2)
pl.figure('Clamping to Middle (f(x)=(100.0/255.0)*x+100)')
pl.gray()
pl.imshow(im3)
pl.figure('Quadratic Transformation (f(x)=255.0*(x/255.0)^2)')
pl.gray()
pl.imshow(im4)

pl.show()