import numpy as np
import legume

%load_ext autoreload
%autoreload 2

# Initialize a lattice (can be 'square', 'hexagonal', or defined by primitive vectors)
lattice = legume.Lattice('hexagonal')

# Initialize a layer with background permittivity 2
layer = legume.ShapesLayer(lattice, eps_b=2)

# Create a square and use the `add_shape` method to add it to the layer
square = legume.Square(eps=10, x_cent=0, y_cent=0, a=0.3)
layer.add_shape(square)

# Create a circle and also add it to the layer
circle = legume.Circle(eps=6, x_cent=0.5, y_cent=0.3, r=0.2)
layer.add_shape(circle)

# Use an in-built visualization method to plot the contours of the shapes we have so far
legume.viz.shapes(layer)