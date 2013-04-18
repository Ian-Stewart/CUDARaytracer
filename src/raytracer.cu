//Ian Stewart & Alexander Newman
//CUDA/OpenGL ray tracer

//Include standard headers
#include <stdio.h>
#include <stdlib.h>
//Include X
#include <X11/X.h>
#include <X11/Xlib.h>
//Include SDL
#include <SDL/SDL.h>
//Include CUDA
#include <cuda.h>
#include <cuda_runtime.h>
//Include project headers
#include "raytraceutils.h"
#include "raystructs.h"
#include "raytracer.h"

#define WIDTH 	500
#define HEIGHT 	500
#define DEPTH 	32

int mouse_old_x;//Old mouse position
int mouse_old_y;

//Defined below main
void DrawScreen(SDL_Surface* screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);

__global__ void test_vbo_kernel(Color3f *c){
	c->r = 0.5;
	c->g = 0.5;
	c->b = 0.5;
}	

int main(int argc, char *argv[]){
	SDL_Surface *screen;
	SDL_Event event;
	
	int keypress = 0;
	
	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		return 1;
	}
	
	if(!(screen = SDL_SetVideoMode(WIDTH, HEIGHT, DEPTH, SDL_HWSURFACE))){
		SDL_Quit();
		return 1;
	}
	
	while(!keypress){
		DrawScreen(screen);
		while(SDL_PollEvent(&event)){
			switch(event.type){
				case SDL_QUIT:
					keypress = 1;
					break;
				case SDL_KEYDOWN:
					keypress = 1;
					break;
			}//End switch(event.type)
		}//End while(SDL_PollEvent)
	}//End while(!keypress)
	
	SDL_Quit();
	return 0;
}

void DrawScreen(SDL_Surface* screen){
	int y = 0;
	int x = 0;
	if(SDL_MUSTLOCK(screen)){
		if(SDL_LockSurface(screen)){
			return;
		}
	}
	
	for(y = 0; y < screen->h;y++){
		for(x = 0; x < screen->w;x++){
			//setpixel(SDL_Surface, x, y, r, g, b)
			setpixel(screen, x, y, 127, 127, 127);
		}
	}//End for(y..){
}

void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b){
    Uint32 *pixmem32;
    Uint32 colour;  
 
    colour = SDL_MapRGB( screen->format, r, g, b );
  
    pixmem32 = (Uint32*) screen->pixels  + y + x;
    *pixmem32 = colour;
}