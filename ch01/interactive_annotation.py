from PIL import Image
import pylab as pl

im = pl.array(Image.open('images/empire.jpg'))
pl.imshow(im)

print('Please click 3 points')

x= pl.ginput(3)
print('You clicked: ' + str(x))
pl.show()