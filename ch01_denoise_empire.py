from PIL import Image
import pylab
import imtools

im = pylab.array(Image.open('empire.jpg').convert('L'))
U,T = imtools.denoise(im, im)

pylab.figure('De-Noised Empire State Building')
pylab.gray()
pylab.imshow(U)
pylab.axis('equal')
pylab.axis('off')
pylab.show()