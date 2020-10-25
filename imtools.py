import os
from PIL import Image
from numpy import *

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# Resize an image array using PIL
def imresize(im, sz):
    pil_im = Image.fromarray(uint8(im))

    return array(pil_im.resize(sz))

# Histogram Equalization
def histeq(im, nbr_bins=256):
    # get image histogram
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

# Compute the average of a list of images
def compute_average(imlist):
    
    # open first image and make into an array of type float.
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')

    averageim /= len(imlist)
    
    # return average as uint8
    return array(averageim, 'uint8')

# De-noise an image
# Implementation of the Rudin-Osher-Fatemi (ROF) denoising model
# using the numerical procedure presented in eq (11) A. Chambolle (2005) 
def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):

    m,n = im.shape # size of noisy image

    # initialize
    U = U_init
    Px = im # x-component to the dual field
    Py = im # y-component of the dual field
    error = 1

    while(error > tolerance):
        Uold = U

        # gradient of primal variable
        GradUx = roll(U,-1,axis=1)-U # x-component of U's gradient
        GradUy = roll(U,-1,axis=0)-U # y-component of U's gradient

        # update the dual variable
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew # update of x-component (dual)
        Py = PyNew/NormNew # update of y-component (dual)

        # update the primal variable
        RxPx = roll(Px,1,axis=1) # right x-translation of x-component
        RyPy = roll(Py,1,axis=0) # right y-translation of y-component

        DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field

        U = im + tv_weight*DivP # update of the primal variable

        # update of error
        error = linalg.norm(U-Uold)/sqrt(n*m)

    return U,im-U # denoised image and texture residual

