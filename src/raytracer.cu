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
#include "raystructs.h"
#include "raytracer.h"
#include "camera.h"
#include "raytraceutils.h"


#define WIDTH 	1000
#define HEIGHT 	1000
#define DEPTH 	32

__global__ void get_camera_rays(Ray *d_crays, Camera *d_camera, int width, int height);
__device__ void CUDA_VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__device__ void CUDA_ScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s);
__device__ void CUDA_InitVector(Vector3f *v, float ix, float iy, float iz);
__device__ void CUDA_Normalize(Vector3f *v);

//Defined below main
void DrawScreen(SDL_Surface* screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);

int mouse_old_x;//Old mouse position
int mouse_old_y;
int width = WIDTH;
int height = HEIGHT;

Camera camera;

__global__ void test_vbo_kernel(Color3f *c){
	c->r = 0.5;
	c->g = 0.5;
	c->b = 0.5;
}

//Set up camera rays for ray tracer
//d_crays is the memory where the camera rays will go
//d_camera is the location of the camera struct in device memory
__global__ void get_camera_rays(Ray *d_crays, Camera *d_camera, int width, int height){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float x;
	float y;
	x = ((float)i)/((float)width);
	y = ((float)j)/((float)height);
	Vector3f direction;
	CUDA_InitVector(&direction, 0, 0, 0);
	CUDA_ScaleAdd(&direction, &(d_camera->across), &(d_camera->corner), x);
	CUDA_ScaleAdd(&direction, &(d_camera->up), &direction, y);
	CUDA_VectorSub(&direction, &direction, &(d_camera->center));
	CUDA_Normalize(&direction);
	d_crays[(j*width) + i].o = d_camera->center;
	d_crays[(j*width) + i].d = direction;
}

__device__ void CUDA_Normalize(Vector3f *v){
	float magnitude = sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));//Length of vector v
	v->x = (v->x)/magnitude;
	v->y = (v->y)/magnitude;
	v->z = (v->z)/magnitude;
}

//v = v1-v2
__device__ void CUDA_VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2){
	v->x = (v1->x) - (v2->x);
	v->y = (v1->y) - (v2->y);
	v->z = (v1->z) - (v2->z);
}

//Sets a vector to some value
__device__ void CUDA_InitVector(Vector3f *v, float ix, float iy, float iz){
	v->x = ix;
	v->y = iy;
	v->z = iz;
}

//CUDA scaleadd v = s*v1 + v2
__device__ void CUDA_ScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s){
	Vector3f v;
	v.x = s*(v1->x);
	v.y = s*(v1->y);
	v.z = s*(v1->z);
	
	v.x += v2->x;
	v.y += v2->y;
	v.z += v2->z;
	
	v0->x = v.x;
	v0->y = v.y;
	v0->z = v.z;
}

int main(int argc, char *argv[]){
	dim3 threadsPerBlock(20,20);//Number of threads per block
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);
	
	
	
	SDL_Surface *screen;
	SDL_Event event;
	
	int keypress = 0;
	
	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		return 1;
	}
	
	if(!(screen = SDL_SetVideoMode(width, height, DEPTH, SDL_HWSURFACE))){
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