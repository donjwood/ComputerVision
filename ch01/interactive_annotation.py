from PIL import Image
import pylab as pl
import numpy as np

im = np.array(Image.open('images/empire.jpg'))
pl.imshow(im)

print('Please click 3 points')

x= pl.ginput(3)
print('You clicked: ' + str(x))
pl.show()