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
__global__ void raytrace(Color3f *d_CUDA_Output, Scene *d_scene, int w, int h, int c);//This actually does the raytracing

__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y);
__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int intersectScene(Scene *scene, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int triangleIntersect(Triangle *triangle, Ray *ray, HitRecord *hit, float tmin, float tmax);

__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u);
__host__ __device__ float findDeterminant(Vector3f *col0, Vector3f *col1, Vector3f *col2);

__host__ __device__ inline void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ inline void InitColor(Color3f *c, float ir, float ig, float ib);

//__host__ __device__ void getShadingColor();
__host__ __device__ void VectorAdd(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void setNormalOfTriangle(Triangle *triangle);
__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void ScaleAdd(Vector3f *v0, float s, Vector3f *v1, Vector3f *v2);
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
void* d_scene;//Device scene pointer
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
	InitVector(&eye, -1, 0, 0);
	InitVector(&at, 0, 0, 0);
	InitVector(&up, 0,0,1);
	initCamera(&camera, &eye, &up, &at, 45, 1);//Set up camera

	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
	
	SDL_Surface *screen;
	SDL_Event event;
	
	int keypress = 0;
	
	int c = 0;//For basic animation
	
	//Scene. Will be replaced with .obj loading later
	Scene scene;
	scene.meshcount = 0;
	scene.spherecount = 3;
	scene.planecount = 1;
	scene.lightcount = 1;
	
	scene.spheres = (Sphere *)malloc(sizeof(Sphere) * 3);//Scene will have three spheres
	scene.meshes = (TriMesh *)malloc(sizeof(TriMesh) * 3);//Three trimeshes
	scene.planes = (Plane *)malloc(sizeof(Plane) * 1);//One plane
	scene.lights = (PointLight *)malloc(sizeof(PointLight) * 1);//One light
	
	InitVector(&(scene.spheres[0].center), 0, 0, 0);
	InitVector(&(scene.spheres[1].center), 3, -1, 1);
	InitVector(&(scene.spheres[2].center), -1, 1, -1);
	
	scene.spheres[0].radius = 1;
	scene.spheres[1].radius = 0.75;
	scene.spheres[2].radius = 1.5;
	
	InitVector(&(scene.planes[0].p), 0, 0, -4);
	InitVector(&(scene.planes[0].normal), 0, 1, 1);
	Normalize(&(scene.planes[0].normal));
	
	InitVector(&(scene.lights[0].pos), 0, 4, 8);
	InitColor(&(scene.lights[0].intensity), 100,100,100);
	
	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		return 1;
	}
	
	if(!(screen = SDL_SetVideoMode(width, height, DEPTH, SDL_HWSURFACE))){
		SDL_Quit();
		return 1;
	}
	while(!keypress){
		//Launch Kernel
		raytrace<<<numBlocks, threadsPerBlock>>>((Color3f *)d_CUDA_Output, (Camera *)d_camera, WIDTH, HEIGHT, c++);
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
__global__ void raytrace(Color3f *d_CUDA_Output, Scene *d_scene, int w, int h, int c){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	Ray cameraRay;
	InitVector(&(cameraRay.d), 1, 0, 0);
	float x;
	float y; //(x,y) is the coordinate for this point in the image such that 0<=x,y<=1
	
	x = (float) i/ (float) w;
	y = (float) j/ (float) h;
	getCameraRay(&cameraRay, &(d_scene->camera), x, y);//Get the camera ray
	
	float tmin = 0.001;
	float tmax = 1000;
	HitRecord hit;

	d_CUDA_Output[(j * w) + i].r = 0;
	d_CUDA_Output[(j * w) + i].g = 0;
	d_CUDA_Output[(j * w) + i].b = 0;
	
	if(intersectScene(d_scene, &cameraRay, &hit, tmin, tmax) == 1){//Ray hit sphere
		tmax = hit.t;//Update t
	}
}

//Given a ray and a scene, find the closest hiting point
__host__ __device__ int intersectScene(Scene *scene, Ray *ray, HitRecord *hit, float tmin, float tmax){
	
}

__host__ __device__ int triangleIntersect(Triangle *triangle, Ray *ray, HitRecord *hit, float tmin, float tmax){
	float a, b;//Barycentric alpha, beta
	
	Vector3f p2subp0 = triangle->p2;//p2-p0
	Vector3f p2subp1 = triangle->p2;//p2-p1
	Vector3f p2subo = triangle->p2;//p2-o
	VectorSub(&p2subp0, &p2subp0, &(triangle->p0));
	VectorSub(&p2subp1, &p2subp1, &(triangle->p1));
	VectorSub(&p2subo, &p2subo, &(ray->o));
	
	float detOfDenom;//Represents the common denominator in the cramer's rule determinant ({{a,b,c},{d,e,f},{g,h,i}})
	detOfDenom = findDeterminant(&(ray->d), &p2subp0, &p2subp1);
	
	if(detOfDenom == 0){//Ray is parallel to triangle
		return 0;
	}
	
	float t = findDeterminant(&p2subo, &p2subp0, &p2subp1)/detOfDenom;
	
	if(t > tmax || t < tmin){//t is out of bounds
		return 0;
	}
	
	a = findDeterminant(&(ray->d), &p2subo, &p2subp1)/detOfDenom;
	b = findDeterminant(&(ray->d), &p2subp0, &p2subo)/detOfDenom;
	
	if(a < 0 || b < 0 || a + b > 1){//Invalid barycentric coordinates - point is outside of triangle
		return 0;
	}
	
	//Now find coordinates of hit
	hit->t = t;
	hit->normal = triangle->normal;
	
	Vector3f p0contrib = triangle->p0;
	Vector3f p1contrib = triangle->p1;
	Vector3f p2contrib = triangle->p2;
	
	VectorScale(&p0contrib, a);
	VectorScale(&p1contrib, b);
	VectorScale(&p2contrib, (1-a-b));
	
	hit->pos = p0contrib;
	VectorAdd(&(hit->pos), &(hit->pos), &p1contrib);
	VectorAdd(&(hit->pos), &(hit->pos), &p2contrib);
	
	hit->material = triangle->material;
	return 1;
}

//Find the intersection of a sphere and a ray, if it exists
__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax){
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
		if(t1 < tmin){
			t1 = t2;
		}
		if(t2 < tmin){
			t2 = t1;
		}
		//Now find smaller t
		if(t1 <= t2){
			hit->t = t1;
		}
		if(t2 < t1){
			hit->t = t2;
		}
		
		if(hit->t > tmax || hit->t < tmin){//Hit is out of bounds
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

//Used when setting up a trimesh. Given three points, finds the normal
__host__ __device__ void setNormalOfTriangle(Triangle *triangle){
	Vector3f v1;
	Vector3f v2;
	//v1 = p1 - p0
	v1.x = triangle->p1.x - triangle->p0.x;
	v1.x = triangle->p1.y - triangle->p0.y;
	v1.x = triangle->p1.z - triangle->p0.z;
	//v2 = p2 - p0
	v2.x = triangle->p2.x - triangle->p0.x;
	v2.x = triangle->p2.y - triangle->p0.y;
	v2.x = triangle->p2.z - triangle->p0.z;
	CrossProduct(&(triangle->normal), &v1, &v2);
	Normalize(&(triangle->normal));
}

//Find the intersection of a ray and plane, if it exists
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float tmin, float tmax){
	Vector3f temp;
	temp = plane->p;
	VectorSub(&temp, &temp, &(ray->o));
	float denom = VectorDot(&(ray->d), &(plane->normal));
	if(denom == 0){//Ray is parallel to plane
		return 0;
	}
	float t = VectorDot(&temp, &(plane->normal)) / denom;
	if(t < tmin || t > tmax){//Hit is out of bounds
		return 0;
	}
	PointOnRay(t, ray, &(hit->pos));//Find the intersection point
	hit->t = t;
	hit->material = plane->material;//Set material of hit
	hit->normal = plane->normal;//Normal is always the same
	Normalize(&(hit->normal));//Should be normalized. Can't assume though...
	return 1;
}

//Given three columns representing a matrix, finds the determinant
__host__ __device__ float findDeterminant(Vector3f *col0, Vector3f *col1, Vector3f *col2){
	return 
	(col0->x*(col1->y*col2->z - col1->z*col2->y)) 
	- (col1->x*(col0->y*col2->z - col0->z*col2->y))
	+ (col2->x*(col0->y*col1->z - col0->z*col1->y));
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

//Find a point on a ray given some t and a ray and load that point into pos
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

//v = v1 + v2
__host__ __device__ void VectorAdd(Vector3f *v, Vector3f *v1, Vector3f *v2){
	v->x = v1->x + v2->x;
	v->y = v1->y + v2->y;
	v->z = v1->z + v2->z;
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
	
	camera->across = U;
	VectorScale(&(camera->across), right-left);
	
	camera->up = V;
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