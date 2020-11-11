from PIL import Image
import pylab as pl
import numpy as np

im = np.array(Image.open('data/empire.jpg'))

pl.imshow(im)
x = [100,100,400,400]
y = [200,500,200,500]

pl.plot(x,y,'r*')
pl.plot(x[:2], y[:2])
pl.title('Plotting: "empire.jpg"')
pl.show()