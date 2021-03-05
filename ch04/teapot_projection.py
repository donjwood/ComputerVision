from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import pickle
from common import ar

width,height = 1000,747

def setup():
    """ Setup window and pygame environment. """
    pygame.init()
    pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')

# load camera data
with open('pickle/ar_camera.pkl','rb') as f:
    K = pickle.load(f)
    Rt = pickle.load(f)

setup()
ar.draw_background('data/book_perspective.bmp', width, height)
ar.set_projection_from_camera(K, width, height)
ar.set_modelview_from_camera(Rt)
ar.draw_teapot(0.10)

while True:
    event = pygame.event.poll()
    if event.type in (QUIT,KEYDOWN):
        break
    pygame.display.flip()