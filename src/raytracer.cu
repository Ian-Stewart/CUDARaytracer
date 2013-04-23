//Ian Stewart & Alexander Newman
//CUDA/SDL ray tracer

//Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define WIDTH 	1000
#define HEIGHT 	1000
#define DEPTH 	32

#ifndef PI
#define PI           3.14159265358979323846
#endif

__global__ void get_camera_rays(Ray *d_crays, Camera *d_camera, int width, int height);
__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void ScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s);
__host__ __device__ void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ void Normalize(Vector3f *v);
__host__ __device__ void VectorScale(Vector3f *v, float s);
__host__ __device__ void Negate(Vector3f *v);
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2);

//Defined below main
void DrawScreen(SDL_Surface* screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);
void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio);


int mouse_old_x;//Old mouse position
int mouse_old_y;
int width = WIDTH;
int height = HEIGHT;

Camera camera;

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

__global__ void test_vbo_kernel(Color3f *c){
	c->r = 0.5;
	c->g = 0.5;
	c->b = 0.5;
}

//Set up camera rays for ray tracer
//d_crays is the memory where the camera rays will go
//d_camera is the location of the camera struct in device memory
__global__ void get_camera_rays(Ray *d_crays, Camera *d_camera, int w, int h){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float x;
	float y;
	x = ((float)i)/((float)w);
	y = ((float)j)/((float)h);
	Vector3f direction;
	InitVector(&direction, 0, 0, 0);
	ScaleAdd(&direction, &(d_camera->across), &(d_camera->corner), x);
	ScaleAdd(&direction, &(d_camera->up), &direction, y);
	VectorSub(&direction, &direction, &(d_camera->center));
	Normalize(&direction);
	d_crays[(j*w) + i].o = d_camera->center;
	d_crays[(j*w) + i].d = direction;
}


//Compute the cross product of a vector
//v1 x v2 = |{{i,j,k},{v1.x,v1.y,v1.z},{v2.x,v2.y,v2.z}}|
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2){
out->x = (v1->y * v2->z) - (v1->z * v2->z);
out->y = -(v1->x * v2->z) - (v1->z * v2->x);
out->z = (v1->x * v2->y) - (v1->y * v2->x);
}

//Negates a vector
__host__ __device__ void Negate(Vector3f *v){
	v->x = -(v->x);
	v->y = -(v->y);
	v->z = -(v->z);
}

//Scales a vector v = s*v
__host__ __device__ void VectorScale(Vector3f *v, float s){
	v->x = s*(v->x);
	v->y = s*(v->y);
	v->z = s*(v->z);
}

//sets v = v/|v|
__host__ __device__ void VectorNormalize(Vector3f *v){
	float magnitude = sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));//Length of vector v
	v->x = (v->x)/magnitude;
	v->y = (v->y)/magnitude;
	v->z = (v->z)/magnitude;
}

__host__ __device__ void Normalize(Vector3f *v){
	float magnitude = sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));//Length of vector v
	v->x = (v->x)/magnitude;
	v->y = (v->y)/magnitude;
	v->z = (v->z)/magnitude;
}

//v = v1-v2
__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2){
	v->x = (v1->x) - (v2->x);
	v->y = (v1->y) - (v2->y);
	v->z = (v1->z) - (v2->z);
}

//Sets a vector to some value
__host__ __device__ void InitVector(Vector3f *v, float ix, float iy, float iz){
	v->x = ix;
	v->y = iy;
	v->z = iz;
}

//scaleadd v = s*v1 + v2
__host__ __device__ void ScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s){
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

void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio){
	//Update camera information
	camera->eye = *in_eye;
	camera->up = *in_up;
	camera->at = *in_at;
	camera->fovy = in_fovy;
	camera->aspect_ratio = ratio;
	
	//Compute points of image plane
	float dist = 1;
	float top = dist * (tanf((camera->fovy * PI)/360));
	float bottom = -top;
	float right = ratio*top;
	float left = right;
	
	Vector3f gaze;
	InitVector(&gaze, 0, 0, 0);
	VectorSub(&gaze, &(camera->at), &(camera->eye));//gaze = at-eye
	
	camera->center = camera->eye;
	Vector3f W = gaze;
	Negate(&W);
	Normalize(&W);
	Vector3f V = camera->up;
	Vector3f U;
	InitVector(&U, 0, 0, 0);
	CrossProduct(&U, &V, &W);//U = VxW
	Normalize(&U);
	CrossProduct(&V, &W, &U);
	
	InitVector(&(camera->corner), 0, 0, 0);
	ScaleAdd(&(camera->corner), &U, &(camera->center), left);
	ScaleAdd(&(camera->corner), &V, &(camera->corner), bottom);
	ScaleAdd(&(camera->corner), &W, &(camera->corner), -dist);
	
	InitVector(&(camera->across),U.x, U.y, U.z);
	VectorScale(&(camera->across), right-left);
	
	InitVector(&(camera->up), V.x, V.y, V.z);
	VectorScale(&(camera->up), top-bottom);
}

void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b){
    Uint32 *pixmem32;
    Uint32 colour;  
 
    colour = SDL_MapRGB( screen->format, r, g, b );
  
    pixmem32 = (Uint32*) screen->pixels  + y + x;
    *pixmem32 = colour;
}