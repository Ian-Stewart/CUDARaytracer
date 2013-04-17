CUDARaytracer
=============

Final project for CS 473 Computer Graphics. Raytracer written in CUDA/OpenGL

Not complete. Still in very early stages.

To compile

nvcc raytracer.cu -lglut -lGL -lGLU -o ../bin/raytracer

I could probably do a makefile....

Note: nvcc does not like gcc 4.6+. To successfully compile, you will need to install an older version of gcc or modify your cuda host_config.h from this
[...]
'#error -- unsupported GNU version! gcc 4.7 and up are not supported!'
[...]

to this
[...]
'//#error -- unsupported GNU version! gcc 4.7 and up are not supported!
#warning -- unsupported GNU version! gcc 4.7 and up are not supported!'
[...]
