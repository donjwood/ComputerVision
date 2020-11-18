"""
White House image matching from the book. Images used were manually 
downloaded from the Google image search since the Panaramio service
doesn't seem to exist anymore.
"""
import os
from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import sift
import pydot

download_path = "data/WhiteHouse"
thumbnail_path = os.getcwd() + "/graphs/wh_thumbnails/"

# list of downloaded filenames
im_paths = imtools.get_imlist(download_path)
nbr_images = len(im_paths)

im_list = []

for path in im_paths:
    im = np.array(Image.open(path).convert("L"))
    im_list.append(imtools.imresize(im,(int(im.shape[1]/2),int(im.shape[0]/2))))


matchscores = np.zeros((nbr_images,nbr_images))

for i in range(nbr_images):
    for j in range(i,nbr_images): # only compute upper triangle
        print('comparing ', im_paths[i], im_paths[j])
        l1,d1 = sift.detect_and_compute(im_list[i])
        l2,d2 = sift.detect_and_compute(im_list[j])
        matches = sift.match_twosided(d1,d2)
        nbr_matches = np.sum(matches > 0)
        print('number of matches = ', nbr_matches)
        matchscores[i,j] = nbr_matches

# copy values
for i in range(nbr_images):
    for j in range(i+1,nbr_images): # no need to copy diagonal
        matchscores[j,i] = matchscores[i,j]

threshold = 2 # min number of matches needed to create link

g = pydot.Dot(graph_type='graph') # don't want the default directed graph

for i in range(nbr_images):
    for j in range(i+1,nbr_images):
        if matchscores[i,j] > threshold:
            # first image in pair
            im = Image.open(im_paths[i])
            im.thumbnail((100,100))
            filename = thumbnail_path+str(i)+'.png'
            im.save(filename) # need temporary files of the right size
            g.add_node(pydot.Node(str(i),fontcolor='transparent',shape='rectangle',image=filename))

            # second image in pair
            im = Image.open(im_paths[j])
            im.thumbnail((100,100))
            filename = thumbnail_path+str(j)+'.png'
            im.save(filename) # need temporary files of the right size
            g.add_node(pydot.Node(str(j),fontcolor='transparent',shape='rectangle',image=filename)) 

            g.add_edge(pydot.Edge(str(i),str(j)))

g.write_png('graphs/whitehouse.png')