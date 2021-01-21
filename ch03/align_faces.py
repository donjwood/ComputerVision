"""
Align the face images.
"""
from common import imregistration

# load the location of control points
xmlFileName = 'data/jkfaces.xml'
points = imregistration.read_points_from_xml(xmlFileName)

# register
imregistration.rigid_alignment(points,'data/jkfaces/')
