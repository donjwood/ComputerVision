import os
from PIL import Image
import numpy as np
from scipy.ndimage import filters

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# Resize an image array using PIL
def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))

    return np.array(pil_im.resize(sz))

# Histogram Equalization
def histeq(im, nbr_bins=256):
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

# Perform gaussian blur
# Works on both color and grayscale images
def gaussian_blur(im, sigma):
    
    if len(im.shape) == 3:
        im_blur = np.empty_like(im)
        for i in range(3):
            im_blur[:,:,i] = filters.gaussian_filter(im[:,:,i],sigma)
            im_blur = np.uint8(im_blur)
    else:
        im_blur = filters.gaussian_filter(im,sigma)
    return im_blur

# Compute the average of a list of images
def compute_average(imlist):
    
    # open first image and make into an array of type float.
    averageim = np.array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '...skipped')

    averageim /= len(imlist)
    
    # return average as uint8
    return np.array(averageim, 'uint8')

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
        GradUx = np.roll(U,-1,axis=1)-U # x-component of U's gradient
        GradUy = np.roll(U,-1,axis=0)-U # y-component of U's gradient

        # update the dual variable
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew # update of x-component (dual)
        Py = PyNew/NormNew # update of y-component (dual)

        # update the primal variable
        RxPx = np.roll(Px,1,axis=1) # right x-translation of x-component
        RyPy = np.roll(Py,1,axis=0) # right y-translation of y-component

        DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field

        U = im + tv_weight*DivP # update of the primal variable

        # update of error
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m)

    return U,im-U # denoised image and texture residual

