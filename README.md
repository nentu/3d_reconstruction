The goal of this project is to understand how visual adometry algorithms work and implement them myself. To do this, I'm going through it piece by piece.

First, I figured out the intrinsic camera matrix and learned how to render 3D models on the screen without third-party 3D libraries

Then I decided to implement a stereo pair. More precisely, to start restoring the 3D coordinates of a model if there are photos of it from different places. The camera coordinates are known

Then I will learn how to restore the 3D coordinates of a model when the camera coordinates are unknown.

And after that, I will connect a physical camera and using ORB detector implement visual adometry