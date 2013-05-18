CUDA Raytracer
=============

Final project for CS 473 Computer Graphics. Raytracer written in CUDA/SDL

Functional but very ugly

To compile (from terminal in root of project directory)

    nvcc ./src/raytracer.cu -o ./bin/raytracer -lSDL -arch sm_30

-lSDL is for SDL graphics

-arch sm_30 is required for recursion

Note: nvcc does not like gcc 4.6+. To successfully compile, you will need to install an older version of gcc or modify your cuda host_config.h. To get nvcc to work, change the line

    #error -- unsupported GNU version! gcc 4.7 and up are not supported!

to this

    //#error -- unsupported GNU version! gcc 4.7 and up are not supported!
    #warning -- unsupported GNU version! gcc 4.7 and up are not supported!

This probably isn't the best way to resolve this issue, but it seems to work well enough.
