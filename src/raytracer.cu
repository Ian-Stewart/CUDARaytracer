//Ian Stewart & Alexander Newman
//CUDA/SDL ray tracer

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <SDL/SDL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "raystructs.h"
#include "raytracer.h"


#ifndef PI
#define PI           3.14159265358979323846
#endif
#define WIDTH 		1000
#define HEIGHT 		1000
#define DEPTH 		32
#define MAX_DEPTH	5

//File reading
void createTriArrayFromFile(Triangle **data, int *tricount, char * filename);

//__host__ __device__ indicates a function that is run on both the GPU and CPU
//__global__ indicates a CUDA kernel
__global__ void raytrace(Color3f *d_CUDA_Output, Sphere *d_spheres, Plane *d_planes, Triangle *d_triangles, PointLight *d_lights, Camera *d_camera, int spherecount, int planecount, int lightcount, int tricount, int w, int h, int c);//This actually does the raytracing

__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int intersectScene(Sphere *d_spheres, Plane *d_planes, Triangle *d_triangles, Ray *ray, HitRecord *hit, int spherecount, int planecount, int tricount, float tmin, float tmax);
__host__ __device__ int triangleIntersect(Triangle *triangle, Ray *ray, HitRecord *hit, float tmin, float tmax);

__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u);
__host__ __device__ float findDeterminant(Vector3f *col0, Vector3f *col1, Vector3f *col2);

__host__ __device__ inline void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ inline void InitColor(Color3f *c, float ir, float ig, float ib);

__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, Triangle *d_triangles, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int tricount, int lightcount);
__host__ __device__ void getLight(PointLight *light, Vector3f *p, Vector3f *pos, Vector3f *lightDir, Color3f *c);
__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y);
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *out);
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *out);
__host__ __device__ void VectorAdd(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void ScaleAdd(Vector3f *v0, float s, Vector3f *v1, Vector3f *v2);
__host__ __device__ void Normalize(Vector3f *v);
__host__ __device__ void Scale(Vector3f *v, float s);
__host__ __device__ void Negate(Vector3f *v);
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2);
__host__ __device__ void PointOnRay(float t, Ray *ray, Vector3f *pos);

//Host only
void DrawScreen(SDL_Surface *screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);
void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio);
unsigned int floatToUint(float f);
void setNormalOfTriangle(Triangle *triangle);


Camera camera;
int mouse_old_x;//Old mouse position
int mouse_old_y;
int width = WIDTH;
int height = HEIGHT;
void* d_CUDA_Output;//Device pointer for output
void* h_CUDA_Output;//Host pointer for output

int main(int argc, char *argv[]){
	dim3 threadsPerBlock(20,20);//Number of threads per block
	dim3 numBlocks(WIDTH/threadsPerBlock.x, HEIGHT/threadsPerBlock.y);

	h_CUDA_Output = malloc(sizeof(Color3f) * WIDTH * HEIGHT);//Allocate memory on host for output
	cudaMalloc(&d_CUDA_Output, sizeof(Color3f) * WIDTH * HEIGHT);//Allocate memory on device for output
	//int i = 0;
	//hard-coded camera, for now
	Vector3f eye;
	Vector3f at;
	Vector3f up;
	InitVector(&at, 0, 0, 0);
	InitVector(&up, 1, 0, 0);
	InitVector(&eye, 0, 3, 6);
	initCamera(&camera, &eye, &up, &at, 50, 1);//Set up camera

	SDL_Surface *screen;
	SDL_Event event;
	
	int c = 0;//For basic animation
	int keypress = 0;
	int spherecount = 4;
	int planecount = 6;
	int lightcount = 3;
	int tricount = 0;
	
	Sphere *spheres 	= (Sphere *)	malloc(sizeof(Sphere) * spherecount);
	Plane *planes 		= (Plane *)	malloc(sizeof(Plane) * planecount);
	PointLight *lights 	= (PointLight *)malloc(sizeof(PointLight) * lightcount);
	//Triangle *triangles	= (Triangle *)	malloc(sizeof(Triangle) * tricount);
	
	Triangle *triangles;
	char diamondFile[] = "./Models/BrilliantDiamond.obj";
	createTriArrayFromFile(&triangles, &tricount, diamondFile);
	
	//Lots of hard-coded stuff incoming...
	
	spheres[0].radius = 1;
	spheres[1].radius = 1;
	spheres[2].radius = 1;
	spheres[3].radius = 1;
	
	//Front face
	InitVector(&(planes[0].p), 0, 10, 0);
	InitVector(&(planes[0].normal), 0, -1, 0);
	Normalize(&(planes[0].normal));
	//Left
	InitVector(&(planes[1].p), -10, 0, 0);
	InitVector(&(planes[1].normal), 1, 0, 0);
	Normalize(&(planes[1].normal));
	//Right
	InitVector(&(planes[2].p), 10, 0, 0);
	InitVector(&(planes[2].normal), -1, 0, 0);
	Normalize(&(planes[2].normal));
	//Back
	InitVector(&(planes[3].p), 0, -5, 0);
	InitVector(&(planes[3].normal), 0, 1, 0);
	Normalize(&(planes[3].normal));
	//Bottom
	InitVector(&(planes[4].p), 0, 0, -10);
	InitVector(&(planes[4].normal), 0, 0, 1);
	Normalize(&(planes[4].normal));
	//Top
	InitVector(&(planes[5].p), 0, 0, 10);
	InitVector(&(planes[5].normal), 0, 0, -1);
	Normalize(&(planes[5].normal));
	
	InitVector(&(lights[0].pos), 1, 8, 1);
	InitColor(&(lights[0].intensity), 25,25,25);
	
	InitVector(&(lights[1].pos), 5, -3, 5);
	InitColor(&(lights[1].intensity), 15,12,12);
	
	InitVector(&(lights[1].pos), 5, -3, -5);
	InitColor(&(lights[2].intensity), 10,12,15);
	
	//Test material
	Material m;
	InitColor(&(m.Ka), 1, 1, 1);
	InitColor(&(m.Kd), 1, 1, 1);
	InitColor(&(m.Ks), 0.25, 0.25, 0.25);
	InitColor(&(m.Kr), 0, 0, 0);
	InitColor(&(m.Kt), 0, 0, 0);
	InitColor(&(m.Ie), 0, 0, 0);
	m.phong_exp = 10;
	m.ior = 0;
	
	Material glass;
	InitColor(&(glass.Ka), 0, 0, 0);
	InitColor(&(glass.Kd), 0, 0, 0);
	InitColor(&(glass.Ks), 0, 0, 0);
	InitColor(&(glass.Kr), 0.05, 0.05, 0.05);
	InitColor(&(glass.Kt), 0.95, 0.95, 0.95);
	InitColor(&(glass.Ie), 0, 0, 0);
	glass.phong_exp = 10;
	glass.ior = 1.45;
	
	Material diamond = glass;
	diamond.ior = 2.417;
	InitColor(&(diamond.Kt), 1,1,1);
	
	//InitColor(&(m.Ie), 0.3,0.3,0.3);
	planes[0].material = m;
	planes[1].material = m;
	planes[2].material = m;
	planes[3].material = m;
	planes[4].material = m;
	planes[5].material = m;
	//InitColor(&(m.Ie), 0, 0, 0);

	
	InitColor(&(m.Kd), 1, 0, 0);
	spheres[0].material = m;
	InitColor(&(m.Kd), 0, 1, 0);
	spheres[1].material = m;
	InitColor(&(m.Kd), 0, 0, 1);
	spheres[2].material = m;
	spheres[3].material = glass;
	
	//Set up diamond properties
	int i;
	for(i = 0; i < tricount; i++){
		triangles[i].material = diamond;
	}
	
	//End hard-coded objects
	
	//CUDA memory pointers
	void* d_camera;
	void* d_spheres;
	void* d_planes;
	void* d_triangles;
	void* d_lights;
	
	//Allocate memory on GPU
	cudaMalloc(&d_camera, sizeof(Camera));//Allocate memory for camera on host
	cudaMalloc(&d_spheres, sizeof(Sphere) * spherecount);//Allocate mem for spheres
	cudaMalloc(&d_planes, sizeof(Plane) * planecount);
	cudaMalloc(&d_lights, sizeof(PointLight) * lightcount);//For lights
	cudaMalloc(&d_triangles, sizeof(Triangle) * tricount);//For triangles. No extra structures to speed intersection checking currently implemented
	
	//Begin copying from host to device
	//Copy camera
	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
	//Copy planes
	cudaMemcpy(d_planes, planes, sizeof(Plane) * planecount, cudaMemcpyHostToDevice);
	//Copy triangles
	cudaMemcpy(d_triangles, triangles, sizeof(Triangle) * tricount, cudaMemcpyHostToDevice);
	//Copy lights
	cudaMemcpy(d_lights, lights, sizeof(PointLight) * lightcount, cudaMemcpyHostToDevice);
	
	//End memory copying from host to device
	
	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		return 1;
	}
	
	if(!(screen = SDL_SetVideoMode(width, height, DEPTH, SDL_HWSURFACE))){
		SDL_Quit();
		return 1;
	}
	timeval start, end;//For measuring frame length
	long time;//no see ha ha ha
	while(!keypress){
		gettimeofday(&start, NULL);
		
		//Move camera
		InitVector(&up, cos(M_PI/2 + (float)c/150), 0, -sin(M_PI/2 + (float)c/150));
		Normalize(&up);
		InitVector(&eye, 5*cos((float)-c/150), 3, 5*sin((float)-c/150));
		initCamera(&camera, &eye, &up, &at, 50, 1);//Set up camera
		cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
		//End camera movement
		
		//Move spheres
		InitVector(&(spheres[0].center), 3*cos((float)c/100), 			-2, 	3*sin((float)c/100));
		InitVector(&(spheres[1].center), 3*cos(M_PI/2 + (float)c/100), 		-2, 	3*sin(M_PI/2 + (float)c/100));
		InitVector(&(spheres[2].center), 3*cos(M_PI + (float)c/100), 		-2, 	3*sin(M_PI + (float)c/100));
		InitVector(&(spheres[3].center), 3*cos((3*M_PI)/2 + (float)c/100), 	-2, 	3*sin((3*M_PI)/2 + (float)c/100));
		//End sphere movement
		
		//Copy new sphere data to GPU
		cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * spherecount, cudaMemcpyHostToDevice);

		
		//Launch Kernel
		raytrace<<<numBlocks, threadsPerBlock>>>((Color3f *)d_CUDA_Output, (Sphere *) d_spheres, (Plane *) d_planes, (Triangle *) d_triangles, (PointLight *) d_lights, (Camera *)d_camera, spherecount, planecount, tricount, lightcount, WIDTH, HEIGHT, c++);
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
		gettimeofday(&end, NULL);
		time = (long) (end.tv_usec - start.tv_usec);
		printf("Frame took %lu msec (FPS: %lu)\n", time/1000, 1000000/time);
	}//End while(!keypress)
	
	//Free memory on GPU
	cudaFree(d_camera);
	cudaFree(d_spheres);
	cudaFree(d_lights);
	cudaFree(d_planes);
	cudaFree(d_triangles);
	cudaFree(d_CUDA_Output);
	
	//Free other pointers
	free(spheres);
	free(planes);
	free(lights);
	free(screen);
	free(triangles);
	free(h_CUDA_Output);
	return 0;
}

//Used when setting up a trimesh. Given three points, finds the normal
void setNormalOfTriangle(Triangle *triangle){
	Vector3f v1;
	Vector3f v2;
	//v1 = p1 - p0
	v1.x = triangle->p0.x - triangle->p1.x;
	v1.y = triangle->p0.y - triangle->p1.y;
	v1.z = triangle->p0.z - triangle->p1.z;
	//v2 = p2 - p0
	v2.x = triangle->p0.x - triangle->p2.x;
	v2.y = triangle->p0.y - triangle->p2.y;
	v2.z = triangle->p0.z - triangle->p2.z;
	CrossProduct(&(triangle->n), &v1, &v2);
	Normalize(&(triangle->n));
	/*
	printf("V1: (%f, %f, %f)\nV2: (%f, %f, %f)\nN: (%f, %f, %f)\nP0: (%f, %f, %f)\nP1: (%f, %f, %f)\nP2: (%f, %f, %f)\n",
		v1.x, v1.y, v1.z,
		v2.x, v2.y, v2.z,
		triangle->n.x,triangle->n.y,triangle->n.z,
		triangle->p0.x, triangle->p0.y, triangle->p0.z,
		triangle->p1.x, triangle->p1.y, triangle->p1.z,
		triangle->p2.x, triangle->p2.y, triangle->p2.z
	);*/
}

//Find the intersection of a sphere and a ray, if it exists
__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax){
	Vector3f v;
	Sphere s = *sphere;
	
	Ray r = *ray;
	
	HitRecord h = *hit;
	h.t = tmax;
	h.pos.x = -100;
	h.pos.y = -100;
	h.pos.z = -100;
	h.normal.x = -100;
	h.normal.y = -100;
	h.normal.z = -100;
	
	v.x = r.o.x - s.center.x;
	v.y = r.o.y - s.center.y;
	v.z = r.o.z - s.center.z;
	float t = 0;
	float B = 2*VectorDot(&v, &(r.d));
	float C = VectorDot(&v, &v) - pow(s.radius,2);
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
			t = t1;
		}
		if(t2 < t1){
			t = t2;
		}
		if(t > tmax || t < tmin){//Hit is out of bounds
			return 0;
		}
		//hit->t = t;
		h.t = t;
		
		PointOnRay(t, &r, &(h.pos));//Find the hitting point and set hit->pos to it
		h.material = s.material;//Set hit material
		//Normal at hitting point P is (P-Center)/|(P-Center) or (P-Center) normalized
		InitVector(&(h.normal),
			h.pos.x - s.center.x,
			h.pos.y - s.center.y,
			h.pos.z - s.center.z
		);
		Normalize(&(h.normal));
		*hit = h;
		return 1;
	}//End else / if(discriminant < 0)
}

//Kernel that actually raytraces
//Size of each array of objects is given by 'x'count integers
__global__ void raytrace( Color3f *d_CUDA_Output, Sphere *d_spheres, Plane *d_planes, Triangle *d_triangles, PointLight *d_lights, Camera *d_camera, int spherecount, int planecount, int tricount, int lightcount, int w, int h, int c){
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	float x;
	float y; //(x,y) is the coordinate for this point in the image such that 0<={x, y}<=1
	
	x = (float) i/ (float) w;
	y = (float) j/ (float) h;

	//Used in rendering loop below
	HitRecord 	hits[MAX_DEPTH];//Array of hit records
	Ray 		lightRays[MAX_DEPTH + 1];//Array of rays
	Color3f 	colors[MAX_DEPTH];//Output from iterative ray tracing
	Color3f 	reflColors[MAX_DEPTH];//Output from reflective stuff

	getCameraRay(&lightRays[0], d_camera, x, y);//Get the camera ray
	
	float tmin = 0.001;
	float tmax = 1000;

	d_CUDA_Output[(j * w) + i].r = 0;
	d_CUDA_Output[(j * w) + i].g = 0;
	d_CUDA_Output[(j * w) + i].b = 0;
	Color3f color;
	InitColor(&color, 0, 0, 0);
	//Ray reflectedRay;//Used for 1-level reflection
	//HitRecord reflectedHit;
	
	int rayDepth = 0;
	
	//Set all colors to black and reflectivity/refractive to 0
	for(rayDepth = 0; rayDepth < MAX_DEPTH; rayDepth++){
		InitColor(&colors[rayDepth], 0, 0, 0);
		InitColor(&reflColors[rayDepth], 0, 0, 0);
		InitColor(&(hits[rayDepth].material.Kt), 0, 0, 0);
		InitColor(&(hits[rayDepth].material.Kr), 0, 0, 0);
	}
	
	for(rayDepth = 0; rayDepth < MAX_DEPTH; rayDepth++){
		if(intersectScene(d_spheres, d_planes, d_triangles, &lightRays[rayDepth], &hits[rayDepth], spherecount, planecount, tricount, tmin, tmax) == 1){
			//Iterative refraction
			getShadingColor(&colors[rayDepth], d_spheres, d_planes, d_triangles, d_lights, &lightRays[rayDepth], &hits[rayDepth], spherecount, planecount, tricount, lightcount);
			if(hits[rayDepth].material.Kt.r > 0 || hits[rayDepth].material.Kt.g > 0 || hits[rayDepth].material.Kt.b > 0){//Surface is refractive
			//Generate refracted ray and store it for next iteration
				lightRays[rayDepth + 1].o = hits[rayDepth].pos;
				lightRays[rayDepth + 1].d = lightRays[rayDepth].d;
				Refract(&(lightRays[rayDepth].d), &(hits[rayDepth].normal), hits[rayDepth].material.ior, &(lightRays[rayDepth + 1].d));
			} else {
			rayDepth = MAX_DEPTH;//End recursion
			}
			//Reflective - disabled right now for diamond rendering
			/*
			if(hits[rayDepth].material.Kr.r > 0 || hits[rayDepth].material.Kr.g > 0 || hits[rayDepth].material.Kr.b > 0){//Surface is reflective, do one level of reflection tracing
				reflectedRay.o = hits[rayDepth].pos;
				reflectedRay.d = lightRays[rayDepth].d;
				Negate(&(reflectedRay.d));
				Reflect(&(reflectedRay.d), &(hits[rayDepth].normal), &(reflectedRay.d));
				if(intersectScene(d_spheres, d_planes, &reflectedRay, &reflectedHit, spherecount, planecount, tmin, tmax) == 1){
					getShadingColor(&reflColors[rayDepth], d_spheres, d_planes, d_lights, &reflectedRay, &reflectedHit, spherecount, planecount, lightcount);
				}
			}
			*/
		}
	}
	
	color.r += colors[0].r;
	color.g += colors[0].g;
	color.b += colors[0].b;
	
	for(rayDepth = 0; rayDepth < MAX_DEPTH-1; rayDepth++){
		color.r += colors[rayDepth + 1].r * hits[rayDepth].material.Kt.r + reflColors[rayDepth + 1].r * hits[rayDepth].material.Kr.r;
		color.g += colors[rayDepth + 1].g * hits[rayDepth].material.Kt.g + reflColors[rayDepth + 1].g * hits[rayDepth].material.Kr.g;
		color.b += colors[rayDepth + 1].b * hits[rayDepth].material.Kt.b + reflColors[rayDepth + 1].b * hits[rayDepth].material.Kr.b;
	}
	//Clamp to 1 - causes weird issues if this isn't done
	if(color.r > 1) color.r = 1;
	if(color.g > 1) color.g = 1;
	if(color.b > 1) color.b = 1;
	d_CUDA_Output[(j * w) + i] = color;
}

//Given a ray and a scene, find the closest hiting point
__host__ __device__ int intersectScene(Sphere *spheres, Plane *planes, Triangle *triangles, Ray *ray, HitRecord *hit, int spherecount, int planecount, int tricount, float tmin, float tmax){
	int i;
	int hitSomething = 0;
	HitRecord tempHit;
	for(i = 0; i < spherecount; i++){
		if(sphereIntersect(&(spheres[i]), ray, &tempHit, tmin, tmax) == 1){
			tmax = tempHit.t;
			*hit = tempHit;
			hitSomething = 1;
		}
	}
	//}
	//Check planes
	for(i = 0; i < planecount; i++){//Check to see if ray intersects any planes
		if(planeIntersect(planes + i, ray, &tempHit, tmin, tmax) == 1){//ray intersects with plane
			hitSomething = 1;
			*hit = tempHit;
			tmax = tempHit.t;
		}//endif
	}//end for (i = 0; i < scene->planecount...)
	
	for(i = 0; i < tricount; i++){//Finally check to see if ray intersects any triangles
		if(triangleIntersect(triangles + i, ray, &tempHit, tmin, tmax) == 1){
			hitSomething = 1;
			*hit = tempHit;
			tmax = tempHit.t;
		}
	}
	return hitSomething;
}

//Get the shading color at a hitting point
//Recursively calls itself on reflective and refractive surfaces
__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, Triangle *d_triangles, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int tricount, int lightcount){
	Vector3f lightPos, lightDir, flippedRay, R;
	Ray tempRay;
	HitRecord shadowed;
	int i;
	Color3f tempColor;
	//Color3f shadingColor;//Temporarily store calculated color - prevents double-drawing some shapes and improves render speed
	//InitColor(&shadingColor, 0, 0, 0);
	
	//Iterate through lights to find surface shading color
	for(i = 0; i < lightcount; i++){
		getLight(d_lights + i, &(hit->pos), &lightPos, &lightDir, &tempColor);
		
		//Now check if shadowed
		Normalize(&lightDir);
		
		tempRay.d = lightDir;
		tempRay.o = hit->pos;
		if(intersectScene(d_spheres, d_planes, d_triangles, &tempRay, &shadowed, spherecount, planecount, tricount, 0.01, sqrtf((lightDir.x * lightDir.x) + (lightDir.y * lightDir.y) + (lightDir.z * lightDir.z))) == 0){//No objects blocking the ray, do light calculation
			//Add diffuse portion
			c->r += tempColor.r * hit->material.Kd.r * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			c->g += tempColor.g * hit->material.Kd.g * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			c->b += tempColor.b * hit->material.Kd.b * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			
			c->r += tempColor.r * hit->material.Ks.r * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
			c->g += tempColor.g * hit->material.Ks.g * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
			c->b += tempColor.b * hit->material.Ks.b * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
		}//end if(intersectScene() == 0)
	}//End light shading loop
	//Add in emissive portion of material
	c->r += hit->material.Ie.r;
	c->g += hit->material.Ie.g;
	c->b += hit->material.Ie.b;
		//Add in ambient light portion
		//Not currently implemented
}

//Find the light intensity of a light at a point, and find useful information for shadow calculation
//Input: light, hit position
//Output: pos, lightDir, c
//Pos is the position of the light, lightDir is the direction from the light to p
__host__ __device__ void getLight(PointLight *light, Vector3f *p, Vector3f *pos, Vector3f *lightDir, Color3f *c){
	//light struct - pos, intensity
	*pos = light->pos;
	VectorSub(lightDir, pos, p);//lightDir = pos - p
	//Find light intensity
	//r = length of the vector from hit to light
	float r = 1/((lightDir->x * lightDir->x) + (lightDir->y * lightDir->y) + (lightDir->z * lightDir->z));
	*c = light->intensity;
	c->r = r * c->r;
	c->g = r * c->g;
	c->b = r * c->b;
}

//Refract around a given normal and index of refraction
//Dir is assumed to be pointing into hit point
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *out){
	float mu;
	Vector3f temp;
	if(VectorDot(normal, dir) < 0){
		mu = 1/ior;
	} else {
		mu = ior;
	}
	
	float cos_thetai = VectorDot(dir, normal);
	float sin_thetai2 = 1 - (cos_thetai*cos_thetai);
	
	if(mu*mu*sin_thetai2 > 1){
		return;//Do nothing
	}
	
	float sin_thetar = mu*sqrtf(sin_thetai2);
	float cos_thetar = sqrtf(1 - (sin_thetar * sin_thetar));
	
	temp = *normal;
	
	if(cos_thetai > 0){
		Scale(&temp, (-mu * cos_thetai) + cos_thetar);
		ScaleAdd(&temp, mu, dir, &temp);
	} else {
		Scale(&temp, (-mu * cos_thetai) - cos_thetar);
		ScaleAdd(&temp, mu, dir, &temp);
	}
	
	Normalize(&temp);
	*out = temp;
}

//Find a reflected ray given an incoming ray and a surface normal
//Assumes dir is pointing away from the hit point
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *out){
	*out = *normal;
	Scale(out, 2 * VectorDot(dir, normal));
	VectorSub(out, out, dir);
}

//Find the intersection of a ray and a triangle
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
	hit->normal = triangle->n;
	
	Vector3f p0contrib = triangle->p0;
	Vector3f p1contrib = triangle->p1;
	Vector3f p2contrib = triangle->p2;
	
	Scale(&p0contrib, a);
	Scale(&p1contrib, b);
	Scale(&p2contrib, (1-a-b));
	
	hit->pos = p0contrib;
	VectorAdd(&(hit->pos), &(hit->pos), &p1contrib);
	VectorAdd(&(hit->pos), &(hit->pos), &p2contrib);
	
	hit->material = triangle->material;
	return 1;
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
	Vector3f thrV = *v;
	Vector3f thrU = *u;
	return (thrV.x * thrU.x) + (thrV.y * thrU.y) + (thrV.z * thrU.z);
}

//Compute the cross product of a vector
//v1 x v2 = |{{i,j,k},{v1.x,v1.y,v1.z},{v2.x,v2.y,v2.z}}|
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2){
	out->x = (v1->y * v2->z) - (v1->z * v2->y);
	out->y = -((v1->x * v2->z) - (v1->z * v2->x));
	out->z = (v1->x * v2->y) - (v1->y * v2->x);
}

//Negates a vector v = -v
__host__ __device__ void Negate(Vector3f *v){
	v->x = -(v->x);
	v->y = -(v->y);
	v->z = -(v->z);
}

//Scales a vector v = s*v
__host__ __device__ void Scale(Vector3f *v, float s){
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

//scaleadd v0 = s*v1 + v2
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
	Scale(&(camera->across), right-left);
	
	camera->up = V;
	Scale(&(camera->up), top-bottom);
}

//Converts float 0-1 to 0-255
unsigned int floatToUint(float f){
	unsigned int u = (int)(f*255);
	return u;
}

//Draws the output of the CUDA kernel on the screen
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

void setpixel(SDL_Surface *screen, int x, int iny, Uint8 r, Uint8 g, Uint8 b){
	Uint32 *pixmem32;
	Uint32 colour;  
	int y = iny*HEIGHT;
	colour = SDL_MapRGB( screen->format, r, g, b );

	pixmem32 = (Uint32*) screen->pixels  + y + x;
	*pixmem32 = colour;
}


//=================================
//OBJ parsing - code by Alex Newman
//=================================

typedef struct{
	Triangle * data;
	int length;
}TriArray;

//storage for the faces and vertices
//Cant do an arrayList in C, this is the next best thing for storing a growing array of multiple variable inputs
int **faces;
Vector3f *vertices;
int numberOfFaces;
int numberOfVertices;

//takes a face value and outputs a triangle outputs through the first argument
void triangulate(Triangle *out, int *face){
	out->p0 = vertices[face[0]];
	out->p1 = vertices[face[1]];
	out->p2 = vertices[face[2]];
	setNormalOfTriangle(out);
}

//sets the size of the local int ** for faces
void setFacesSize(){
	faces = (int **)malloc(sizeof(int*) * (numberOfFaces));
	for(int i = 0; i < numberOfFaces; i++){
		faces[i] = (int *)malloc( sizeof(int) * 3 );
	}
}

//sets the size of the local Vector3f * for vertices
void setVerticesSize(){
	vertices = (Vector3f*)malloc( sizeof(Vector3f) * (numberOfVertices + 1));
}

/*read through the file once to determine the number of vertices and faces
after reading through the file once set the size of the local arrays*/
void setFileSize(char filename[]){
	numberOfFaces = 0;
	numberOfVertices = 0;
	//open file
	FILE *file = fopen(filename, "r");
	if(file != NULL){
	  char line [128];
	  //read in line of file
	  while(fgets (line, sizeof line, file) != NULL){
	  	//if the line starts with v it is a vertex
	    if(line[0] == 'v'){
	     	numberOfVertices++;
	    }
	    //if the line starts with f it is a face
	    if(line[0] == 'f'){
	     	numberOfFaces++;
	    }
	  }
	  //close file
	  fclose (file);
	}
	else{
	  perror (filename);
	}
	//set the size of the vertex and face arrays
	setFacesSize();
	setVerticesSize();
}

/*reads through the file specified populating the local arrays with their contents*/ 
void readInObject(char * filename){
	int currentFace = 0;
	int currentVertex = 1;
	FILE *file = fopen(filename, "r");

	if(file != NULL){
		char line [128];
		while(fgets (line, sizeof line, file) != NULL){
			if(line[0] == 'v'){
				//assign values to the array then increment
				//need to go over pointer logic here!!!!!!!!!!!!!!!!!!!!!!!!!!
				sscanf(line, "%*s %f %f %f", &(vertices[currentVertex].x), &(vertices[currentVertex].y), &(vertices[currentVertex].z));
				//printf("%f %f %f\n", vertices[currentVertex].x, vertices[currentVertex].y, vertices[currentVertex].z);
				currentVertex++;
			}
			if(line[0] == 'f'){
				//assign values to the array then increment
				sscanf(line, "%*s %d %d %d", &(faces[currentFace][0]), &(faces[currentFace][1]), &(faces[currentFace][2]));
				//printf("%i - %i %i %i\n", currentFace, faces[currentFace][0], faces[currentFace][1], faces[currentFace][2]);
				currentFace++;
				
			}
		}
	}
	else{
	  perror (filename);
	}
	fclose (file);
}

//returns the lower float
float testMin(float a, float b){
	if(a > b)return b;
	else return a;
}

//returns the higher float
float testMax(float a, float b){
	if(b > a)return b;
	else return a;
}

//constructs two triangles to form the surface of a box, outputs fromt he first 2 arguments
void createSquareFace(Triangle * t1, Triangle * t2, Vector3f v1, Vector3f v2, Vector3f v3, Vector3f v4){
	//to make sure sure the box is formed correctly v1 and v3 must be opposite corners as do v2 and v4
	//this means between v1 and v3 x,y,z do not share any values besides the plane they are on
	t1->p0 = v1;
	t1->p1 = v2;
	t1->p2 = v4;
	//May need to correct pointer logic in this method!!!!!!!!!!!!!!!!!!!1
	//calcNormal(&(t1->n), &v1, &v2, &v4);
	t2->p0 = v3;
	t2->p1 = v4;
	t2->p2 = v2;
	//calcNormal(&(t1->n), &v3, &v4, &v2);
}

//creates a bounding box around the object read in from file
void createBoundingBox(TriArray * boundingBox){
	//first find the max and min values for x,y,z of this object
	int minX = INT_MAX, minY = INT_MAX, minZ = INT_MAX;
	int maxX = 0, maxY = 0, maxZ = 0;
	for(int i = 0; i < numberOfVertices; i++){
		maxX = testMax(vertices[i].x, maxX);
		minX = testMin(vertices[i].x, minX);
		maxY = testMax(vertices[i].y, maxY);
		minY = testMin(vertices[i].y, minY);
		maxZ = testMax(vertices[i].z, maxZ);
		minZ = testMin(vertices[i].z, minZ);
	}

	//next turn those max and min values into the vertices of a box
	Vector3f * bbv = (Vector3f *)malloc(sizeof(Vector3f) * 8);
	InitVector(&bbv[0], minX, minY, minZ);
	InitVector(&bbv[1], maxX, minY, minZ);
	InitVector(&bbv[2], minX, maxY, minZ);
	InitVector(&bbv[3], minX, minY, maxZ);
	InitVector(&bbv[4], maxX, minY, maxZ);
	InitVector(&bbv[5], maxX, maxY, minZ);
	InitVector(&bbv[6], minX, maxY, maxZ);
	InitVector(&bbv[7], maxX, maxY, maxZ);

	//next contruct the box using two triangles per face
	Triangle *bbf = (Triangle *)malloc(sizeof(Triangle) * 12);
	//z plane min
	createSquareFace(&bbf[0], &bbf[1], bbv[0], bbv[1], bbv[5], bbv[2]);
	//z plane max
	createSquareFace(&bbf[2], &bbf[3], bbv[3], bbv[4], bbv[7], bbv[6]);
	//y plane min
	createSquareFace(&bbf[4], &bbf[5], bbv[0], bbv[1], bbv[4], bbv[3]);
	//y plane max
	createSquareFace(&bbf[6], &bbf[7], bbv[2], bbv[5], bbv[7], bbv[6]);
	//x plane min
	createSquareFace(&bbf[8], &bbf[9], bbv[0], bbv[2], bbv[6], bbv[3]);
	//x plane max
	createSquareFace(&bbf[10], &bbf[11], bbv[1], bbv[4], bbv[7], bbv[5]);

	//TriArray boundingBox = malloc(sizeof(TriArray));
	boundingBox->data = bbf;
	boundingBox->length = 12;
	//return boundingBox;
}

//More or less the main of this code calls the other methods constructs a TriArray for the object
//Also creates a bounding box that isn't stored anywhere yet
void createTriArrayFromFile(Triangle **data, int *tricount, char * filename){
	setFileSize(filename);
	//printf("faces %d\n", numberOfFaces);
	//printf("vertices %d\n", numberOfVertices);
	readInObject(filename);
	//printf("read in complete\n");
	Triangle *triangles = (Triangle *)malloc(sizeof(Triangle) * numberOfFaces);
	if(triangles == NULL){printf("triangle malloc failed");}
	for(int i = 0; i < numberOfFaces; i++){
		//printf("before triangluate %d\n", i);
		triangulate(&(triangles[i]), faces[i]);
	}
	printf("Faces: %i Verts: %i\n", numberOfFaces, numberOfVertices);
	*tricount = numberOfFaces;
	*data = triangles;
	free(faces);
	free(vertices);
}