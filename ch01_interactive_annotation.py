from PIL import Image
from pylab import *

im = array(Image.open('empire.jpg'))
imshow(im)

print('Please click 3 points')

x= ginput(3)
print('You clicked: ' + str(x))
show()