from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open('empire.jpg').convert('L'))

im2 = 255 - im # invert image

im3 = (100.0/255) * im + 100 # clamp to interval 100..200

im4 = 255.0 * (im/255.0)**2 # squared

# Output min/max to console
print('Figure 1 min/max: ' + str(im.min()) + '/' + str(im.max()))
print('Figure 2 min/max: ' + str(im2.min()) + '/' + str(im2.max()))
print('Figure 3 min/max: ' + str(im3.min()) + '/' + str(im3.max()))
print('Figure 4 min/max: ' + str(im4.min()) + '/' + str(im4.max()))


figure('Original (f(x)=x)')
gray()
imshow(im)
figure('Inversion (f(x)=255-x)')
gray()
imshow(im2)
figure('Clamping to Middle (f(x)=(100.0/255.0)*x+100)')
gray()
imshow(im3)
figure('Quadratic Transformation (f(x)=255.0*(x/255.0)^2)')
gray()
imshow(im4)

show()