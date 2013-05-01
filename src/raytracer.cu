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

#ifndef PI
#define PI           3.14159265358979323846
#endif

#define WIDTH 	1000
#define HEIGHT 	1000
#define DEPTH 	32

//CUDA functions
//__host__ __device__ indicates a function that is run on both the GPU and CPU
//__global__ indicates a CUDA kernel
__global__ void test_vbo_kernel(Color3f *CUDA_Output, int w, int h);//Purely for testing CUDA color output
__global__ void raytrace(Color3f *d_CUDA_Output, Camera *d_camera, int w, int h);//This actually does the raytracing

__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y);
__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float *tmin, float *tmax);
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float *tmin, float *tmax);

__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u);

__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void ScaleAdd(Vector3f *v0, float s, Vector3f *v1, Vector3f *v2);
__host__ __device__ void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ void InitColor(Color3f *c, float ir, float ig, float ib);
__host__ __device__ void Normalize(Vector3f *v);
__host__ __device__ void VectorScale(Vector3f *v, float s);
__host__ __device__ void Negate(Vector3f *v);
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2);
__host__ __device__ void PointOnRay(float t, Ray *ray, Vector3f *pos);


//Defined below main
void DrawScreen(SDL_Surface *screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);
void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio);
unsigned int floatToUint(float f);

int mouse_old_x;//Old mouse position
int mouse_old_y;
int width = WIDTH;
int height = HEIGHT;

Camera camera;
void* d_CUDA_Output;//Device pointer for output
void* d_camera;//Device camera pointer
void* h_CUDA_Output;//Host pointer for output

int main(int argc, char *argv[]){
	dim3 threadsPerBlock(20,20);//Number of threads per block
	dim3 numBlocks(WIDTH/threadsPerBlock.x, HEIGHT/threadsPerBlock.y);
	
	h_CUDA_Output = malloc(sizeof(Color3f) * WIDTH * HEIGHT);//Allocate memory on host for output
	cudaMalloc(&d_CUDA_Output, sizeof(Color3f) * WIDTH * HEIGHT);//Allocate memory on device for output
	cudaMalloc(&d_camera, sizeof(Camera));//Allocate memory for camera on host
	
	//hard-coded camera, for now
	Vector3f eye;
	Vector3f at;
	Vector3f up;
	InitVector(&eye, -1, 0, 1);
	InitVector(&at, 0, 0, 1);
	InitVector(&up, 0,0,1);
	initCamera(&camera, &eye, &up, &at, 45, 1);//Set up camera

	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
	
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
	/*
	//For testing purposes. Remove from final code.
	Sphere testSphere;
	InitVector(&(testSphere.center), 0.1, 0.25, 0.5);
	testSphere.radius = 1;
	//Plane testPlane;
	//InitVector(&(testPlane.p), 0, 0, -3);
	//InitVector(&(testPlane.normal), 0, 0, 1);
	Material testMaterial;
	//testPlane.material = testMaterial;
	testSphere.material = testMaterial;
	Ray ray;
	InitVector(&(ray.d), 0,0,-1);
	InitVector(&(ray.o), 0,0,3);
	HitRecord hit;
	printf("%i \n",sphereIntersect(&testSphere, &ray, &hit, 0, 100));
	printf("Hit Normal: %f, %f, %f\n", hit.normal.x, hit.normal.y, hit.normal.z);
	printf("t=%f pos=%f,%f,%f\n=======================\n", hit.t, hit.pos.x, hit.pos.y, hit.pos.z);
	printf("Testing PointOnRay\n");
	Vector3f pos;
	PointOnRay(hit.t, &ray, &pos);
	printf("%f,%f,%f\n--------------------\n", pos.x, pos.y, pos.z);
	return 1;
	*/
	while(!keypress){
		//test_vbo_kernel<<<numBlocks, threadsPerBlock>>>((Color3f *)d_CUDA_Output, WIDTH, HEIGHT);//Run kernel
		//Launch Kernel
		raytrace<<<numBlocks, threadsPerBlock>>>((Color3f *)d_CUDA_Output, (Camera *)d_camera, WIDTH, HEIGHT);
		
		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceSynchronize();//Wait for GPU to finish
		cudaMemcpy(h_CUDA_Output, d_CUDA_Output, sizeof(Color3f) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);//Copy results of GPU kernel to host memory
		DrawScreen(screen);//Update the screen
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

//Kernel that actually raytraces
__global__ void raytrace(Color3f *d_CUDA_Output, Camera *d_camera, int w, int h){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	Ray cameraRay;
	InitVector(&(cameraRay.d), 1, 0, 0);
	//getCameraRay(&cameraRay, (Camera *)d_camera,  w, h, i, j);
	float x;
	float y; //(x,y) is the coordinate for this point in the image such that 0<=x,y<=1
	
	x = (float) i/ (float) w;
	y = (float) j/ (float) h;
	getCameraRay(&cameraRay, (Camera *) d_camera, x, y);//Get the camera ray
	
	//For testing purposes. Remove from final code.
	Sphere testSphere;
	InitVector(&(testSphere.center), 5, 0, 0);
	testSphere.radius = 1;
	
	Plane testPlane;
	InitVector(&(testPlane.p), 10, 0, 0);
	InitVector(&(testPlane.normal), -1, 0, 0);
	//Normalize(&(testPlane.normal));
	
	Material testMaterial;
	//InitColor(&(testMaterial.Ka), 0, 0, 0);
	//InitColor(&(testMaterial.Kd), 0, 0, 0);
	//InitColor(&(testMaterial.Ks), 0, 0, 0);
	//InitColor(&(testMaterial.Kr), 0.5, 0.5, 0.5);
	//InitColor(&(testMaterial.Kt), 0.95, 0.95, 0.95);
	//InitColor(&(testMaterial.Ie), 0, 0, 0);
	//testMaterial.phong_exp = 20;
	//testMaterial.ior = 1.5;//Roughly equal to glass
	
	testPlane.material = testMaterial;
	testSphere.material = testMaterial;
	
	float tmin = 0.001;
	float tmax = 1000;
	HitRecord hit;

	d_CUDA_Output[(j * w) + i].r = 1;
	d_CUDA_Output[(j * w) + i].g = 0;
	d_CUDA_Output[(j * w) + i].b = 0;
	
	if(sphereIntersect(&testSphere, &cameraRay, &hit, &tmin, &tmax) == 1){//Ray hit sphere
		tmax = hit.t;//Update t
		//d_CUDA_Output[(j * w) + i].g = 1;
		d_CUDA_Output[(j * w) + i].r = (hit.normal.x);
		d_CUDA_Output[(j * w) + i].g = (hit.normal.y);
		d_CUDA_Output[(j * w) + i].b = (hit.normal.z);
	}
	if(planeIntersect(&testPlane, &cameraRay, &hit, &tmin, &tmax) == 1){//Ray hit plane
		tmax = hit.t;//Update t
		//d_CUDA_Output[(j * w) + i].b = 1;
		d_CUDA_Output[(j * w) + i].r = (hit.normal.x);
		d_CUDA_Output[(j * w) + i].g = (hit.normal.y);
		d_CUDA_Output[(j * w) + i].b = (hit.normal.z);
	}
}

//Find the intersection of a sphere and a ray, if it exists
__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float *tmin, float *tmax){
	Vector3f v;
	InitVector(&v, 
		ray->o.x - sphere->center.x,
		ray->o.y - sphere->center.y,
		ray->o.z - sphere->center.z
	);
	
	float B = 2*VectorDot(&v, &(ray->d));
	float C = VectorDot(&v, &v) - (sphere->radius * sphere->radius);
	float discriminant = sqrtf(B*B - 4*C);
	if(discriminant < 0){//Ray does not intersect sphere
		return 0;
	} else {
		float t1 = (-B + discriminant)/(2);
		float t2 = (-B - discriminant)/(2);
		if(t1 < *tmin){
			t1 = t2;
		}
		if(t2 < *tmin){
			t2 = t1;
		}
		//Now find smaller t
		if(t1 <= t2){
			hit->t = t1;
		}
		if(t2 < t1){
			hit->t = t2;
		}
		
		if(hit->t > *tmax || hit->t < *tmin){//Hit is out of bounds
			return 0;
		}
		
		PointOnRay(hit->t, ray, &(hit->pos));//Find the hitting point and set hit->pos to it
		hit->material = sphere->material;//Set hit material
		//Normal at hitting point P is (P-Center)/|(P-Center) or (P-Center) normalized
		InitVector(&(hit->normal),
			hit->pos.x - sphere->center.x,
			hit->pos.y - sphere->center.y,
			hit->pos.z - sphere->center.z
		);
		Normalize(&(hit->normal));
		return 1;
	}//End else / if(discriminant < 0)
	
}

//Set up camera rays for ray tracer
//d_crays is the memory where the camera rays will go
//d_camera is the location of the camera struct in device memory
__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y){
	Vector3f direction;
	InitVector(&direction, 0, 0, 0);
	ScaleAdd(&direction, x, &(d_camera->across), &(d_camera->corner));
	ScaleAdd(&direction, y, &(d_camera->up), &direction);
	VectorSub(&direction, &direction, &(d_camera->center));
	Normalize(&direction);
	ray->o = d_camera->center;
	ray->d = direction;
}

//Find the intersection of a ray and plane, if it exists
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float *tmin, float *tmax){
	Vector3f temp;
	InitVector(&temp, 0,0,0);
	VectorSub(&temp, &temp, &(ray->o));
	float denom = VectorDot(&(ray->d), &(plane->normal));
	if(denom == 0){//Ray is parallel to plane
		return 0;
	}
	float t = VectorDot(&temp, &(plane->normal)) / denom;
	if(t < *tmin || t > *tmax){//Hit is out of bounds
		return 0;
	}
	PointOnRay(t, ray, &(hit->pos));//Find the intersection point
	hit->t = t;
	hit->material = plane->material;//Set material of hit
	hit->normal = plane->normal;//Normal is always the same
	Normalize(&(hit->normal));//Should be normalized. Can't assume though...
	return 1;
}

//Find a point on a ray given some t and a ray
__host__ __device__ void PointOnRay(float t, Ray *ray, Vector3f *pos){
	pos->x = ray->o.x + (ray->d.x*t);
	pos->y = ray->o.y + (ray->d.y*t);
	pos->z = ray->o.z + (ray->d.z*t);
}

//Find the dot product of a vector
__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u){
	return (v->x * u->x) + (v->y * u->y) + (v->z * u->z);
}

void DrawScreen(SDL_Surface *screen){
	int y = 0;
	int x = 0;
	if(SDL_MUSTLOCK(screen)){
		if(SDL_LockSurface(screen)){
			return;
		}
	}
	
	Color3f *cudaout = (Color3f *)h_CUDA_Output;
	
	for(y = 0; y < HEIGHT;y++){
		for(x = 0; x < WIDTH;x++){
			setpixel(screen, x, y, floatToUint(cudaout[(x * WIDTH) + y].r), floatToUint(cudaout[(x * WIDTH) + y].g), floatToUint(cudaout[(x * WIDTH) + y].b));
		}
	}//End for(y..){
		
	if(SDL_MUSTLOCK(screen)){
		SDL_UnlockSurface(screen);
	}
	SDL_Flip(screen);
}

//Converts float 0-1 to 0-255
unsigned int floatToUint(float f){
	unsigned int u = (int)(f*255);
	return u;
}

//Test kernel for debugging
__global__ void test_vbo_kernel(Color3f *CUDA_Output, int w, int h){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	CUDA_Output[(j * w) + i].r = 0.1;
	CUDA_Output[(j * w) + i].g = 0.25;
	CUDA_Output[(j * w) + i].b = 0.75;
}

//Compute the cross product of a vector
//v1 x v2 = |{{i,j,k},{v1.x,v1.y,v1.z},{v2.x,v2.y,v2.z}}|
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2){
out->x = (v1->y * v2->z) - (v1->z * v2->y);
out->y = -(v1->x * v2->z) - (v1->z * v2->x);
out->z = (v1->x * v2->y) - (v1->y * v2->x);
}

//Negates a vector v = -v
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

//Normalizes a vector (sets v = v/|v|)
__host__ __device__ void Normalize(Vector3f *v){
	float magnitude = sqrtf( pow(v->x,2) + pow(v->y,2) + pow(v->z,2) );//Length of vector v
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

//Sets a color to some value
__host__ __device__ void InitColor(Color3f *c, float ir, float ig, float ib){
	c->r = ir;
	c->g = ig;
	c->b = ib;
}

//scaleadd v = s*v1 + v2
__host__ __device__ void ScaleAdd(Vector3f *v0, float s, Vector3f *v1, Vector3f *v2){
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
	float left = -right;
	
	Vector3f gaze;
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
	ScaleAdd(&(camera->corner), left, &U, &(camera->center));
	ScaleAdd(&(camera->corner), bottom, &V, &(camera->corner));
	ScaleAdd(&(camera->corner), -dist, &W, &(camera->corner));
	
	InitVector(&(camera->across),U.x, U.y, U.z);
	VectorScale(&(camera->across), right-left);
	
	InitVector(&(camera->up), V.x, V.y, V.z);
	VectorScale(&(camera->up), top-bottom);
}

void setpixel(SDL_Surface *screen, int x, int iny, Uint8 r, Uint8 g, Uint8 b){
	Uint32 *pixmem32;
	Uint32 colour;  
	int y = iny*HEIGHT;
	colour = SDL_MapRGB( screen->format, r, g, b );

	pixmem32 = (Uint32*) screen->pixels  + y + x;
	*pixmem32 = colour;
}