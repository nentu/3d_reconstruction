# Goal

Create a proramm for drawing rotation of 3d cube without 3d libs. 

I will use opencv for drawing.

# How it works

At first we define coorninates of cubes vertex. 

Then we apply tansformation to them. In this case it is matrix rotation.

Then project 3d coordinates to virtual plane using intrinsic camera matrix

Draw points in the coordinates

# Additioanl work

Add `Model` class for describing models and loading models from obj.

Add "depht map" - linear interpolation f(X, Y) = Z.