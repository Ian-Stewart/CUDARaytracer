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
#define MAX_DEPTH	4

//__host__ __device__ indicates a function that is run on both the GPU and CPU
//__global__ indicates a CUDA kernel
__global__ void raytrace(Color3f *d_CUDA_Output, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Camera *d_camera, int spherecount, int planecount, int lightcount, int w, int h, int c);//This actually does the raytracing

__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int intersectScene(Sphere *d_spheres, Plane *d_planes, Ray *ray, HitRecord *hit, int spherecount, int planecount, float tmin, float tmax);
__host__ __device__ int triangleIntersect(Triangle *triangle, TriMesh *trimesh, Ray *ray, HitRecord *hit, float tmin, float tmax);

__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u);
__host__ __device__ float findDeterminant(Vector3f *col0, Vector3f *col1, Vector3f *col2);

__host__ __device__ inline void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ inline void InitColor(Color3f *c, float ir, float ig, float ib);

__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int lightcount);
__host__ __device__ void getLight(PointLight *light, Vector3f *p, Vector3f *pos, Vector3f *lightDir, Color3f *c);
__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y);
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *out);
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *out);
__host__ __device__ void VectorAdd(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void setNormalOfTriangle(Triangle *triangle);
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
	InitVector(&eye, -6, -9, 2);
	initCamera(&camera, &eye, &up, &at, 50, 1);//Set up camera

	SDL_Surface *screen;
	SDL_Event event;
	

	int c = 0;//For basic animation
	int keypress = 0;
	//int totalTris = 0;
	//int meshcount = 0;
	int spherecount = 4;
	int planecount = 6;
	int lightcount = 3;
	
	Sphere *spheres 	= (Sphere *)	malloc(sizeof(Sphere) * spherecount);//Scene will have three spheres
	Plane *planes 		= (Plane *)	malloc(sizeof(Plane) * planecount);//One plane
	PointLight *lights 	= (PointLight *)malloc(sizeof(PointLight) * lightcount);//One light
	//TriMesh *meshes 	= (TriMesh *)	malloc(sizeof(TriMesh) * 3);//Three trimeshes
	
	InitVector(&(spheres[0].center), 2, 0, -0.5);
	InitVector(&(spheres[1].center), 0, 0, -0.5);
	InitVector(&(spheres[2].center), -2, 0, -0.5);
	InitVector(&(spheres[3].center), 0, -3, 0.5);
	
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
	InitVector(&(planes[3].p), 0, -10, 0);
	InitVector(&(planes[3].normal), 0, 1, 0);
	Normalize(&(planes[3].normal));
	//Bottom
	InitVector(&(planes[4].p), 0, 0, -2);
	InitVector(&(planes[4].normal), 0, 0, 1);
	Normalize(&(planes[4].normal));
	//Top
	InitVector(&(planes[5].p), 0, 0, 10);
	InitVector(&(planes[5].normal), 0, 0, -1);
	Normalize(&(planes[5].normal));
	
	InitVector(&(lights[0].pos), 0, 0, 7);
	InitColor(&(lights[0].intensity), 25,25,25);
	
	InitVector(&(lights[1].pos), 0, 4, 3);
	InitColor(&(lights[1].intensity), 10,10,15);
	
	InitVector(&(lights[2].pos), 0, -4.2, 2.5);
	InitColor(&(lights[2].intensity), 15,10,10);
	
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
	InitColor(&(glass.Kr), 0, 0, 0);
	InitColor(&(glass.Kt), 1, 1, 1);
	InitColor(&(glass.Ie), 0, 0, 0);
	glass.phong_exp = 10;
	glass.ior = 1.45;
	
	planes[0].material = m;
	planes[1].material = m;
	planes[2].material = m;
	planes[3].material = m;
	planes[4].material = m;
	planes[5].material = m;
	
	InitColor(&(m.Kd), 1, 0, 0);
	spheres[0].material = m;
	InitColor(&(m.Kd), 0, 1, 0);
	spheres[1].material = m;
	InitColor(&(m.Kd), 0, 0, 1);
	spheres[2].material = m;
	
	spheres[3].material = glass;
	//End material

	//CUDA memory
	void* d_camera;
	//void* d_trimeshes;
	void* d_spheres;
	void* d_planes;
	//void* d_triangles;
	void* d_lights;
	
	//TriMesh *h_flattened_triangles;
	
	cudaMalloc(&d_camera, sizeof(Camera));//Allocate memory for camera on host
	//cudaMalloc(&d_trimeshes, sizeof(TriMesh) * meshcount);//Allocate memory for TriMesh structures
	cudaMalloc(&d_spheres, sizeof(Sphere) * spherecount);//Allocate mem for spheres
	cudaMalloc(&d_planes, sizeof(Plane) * planecount);
	cudaMalloc(&d_lights, sizeof(PointLight) * lightcount);//For lights
	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	
	//Begin copying from host to device
	//Copy camera
	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
	//Copy planes
	cudaMemcpy(d_planes, planes, sizeof(Plane) * planecount, cudaMemcpyHostToDevice);
	//Copy spheres
	cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * spherecount, cudaMemcpyHostToDevice);
	//Copy trimesh structs - cannot deep copy data
	//cudaMemcpy(d_trimeshes, meshes, sizeof(Trimesh * meshcount, cudaMemcpyHostToDevice);
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
	long time;
	
	while(!keypress){
		gettimeofday(&start, NULL);
		cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
		InitVector(&eye, 9 * cos((float)c/100), 9 * sin(((float)c )/100), 0);
		InitVector(&up, -sin((float)c/100), 0, 0);
		Normalize(&up);
		initCamera(&camera, &eye, &up, &at, 50, 1);//Set up camera
		//Launch Kernel
		raytrace<<<numBlocks, threadsPerBlock>>>((Color3f *)d_CUDA_Output, (Sphere *) d_spheres, (Plane *) d_planes, (PointLight *) d_lights, (Camera *)d_camera, spherecount, planecount, lightcount, WIDTH, HEIGHT, c++);
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
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
	return 0;
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
__global__ void raytrace( Color3f *d_CUDA_Output, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Camera *d_camera, int spherecount, int planecount, int lightcount, int w, int h, int c){
	
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
	
	
	int rayDepth = 0;
	
	//Set all colors to black and reflectivity/refractive to 0
	for(rayDepth = 0; rayDepth < MAX_DEPTH; rayDepth++){
		InitColor(&colors[rayDepth], 0, 0, 0);
		InitColor(&reflColors[rayDepth], 0, 0, 0);
		InitColor(&(hits[rayDepth].material.Kt), 0, 0, 0);
		InitColor(&(hits[rayDepth].material.Kr), 0, 0, 0);
	}
	
	for(rayDepth = 0; rayDepth < MAX_DEPTH; rayDepth++){
		if(intersectScene(d_spheres, d_planes, &lightRays[rayDepth], &hits[rayDepth], spherecount, planecount, tmin, tmax) == 1){
			getShadingColor(&colors[rayDepth], d_spheres, d_planes, d_lights, &lightRays[rayDepth], &hits[rayDepth], spherecount, planecount, lightcount);
			if(hits[rayDepth].material.Kt.r > 0 || hits[rayDepth].material.Kt.g > 0 || hits[rayDepth].material.Kt.b > 0){//Surface is refractive
			//Generate refracted ray and store it for next iteration
				lightRays[rayDepth + 1].o = hits[rayDepth].pos;
				lightRays[rayDepth + 1].d = lightRays[rayDepth].d;
				Refract(&(lightRays[rayDepth].d), &(hits[rayDepth].normal), hits[rayDepth].material.ior, &(lightRays[rayDepth + 1].d));
			} else {
			rayDepth = MAX_DEPTH;//End recursion
			}
			
			if(hits[rayDepth].material.Kr.r > 0 || hits[rayDepth].material.Kr.g > 0 || hits[rayDepth].material.Kr.b > 0){//Surface is reflective
				
			}
		}
	}
	
	color.r += colors[0].r;
	color.g += colors[0].g;
	color.b += colors[0].b;
	
	for(rayDepth = 0; rayDepth < MAX_DEPTH-1; rayDepth++){
		color.r += colors[rayDepth + 1].r * hits[rayDepth].material.Kt.r;// + reflColors[rayDepth].r * hits[rayDepth].material.Kr.r;
		color.g += colors[rayDepth + 1].g * hits[rayDepth].material.Kt.g;// + reflColors[rayDepth].g * hits[rayDepth].material.Kr.g;
		color.b += colors[rayDepth + 1].b * hits[rayDepth].material.Kt.b;// + reflColors[rayDepth].b * hits[rayDepth].material.Kr.b;
	}
	//Clamp to 1 - causes weird issues if this isn't done
	if(color.r > 1) color.r = 1;
	if(color.g > 1) color.g = 1;
	if(color.b > 1) color.b = 1;
	d_CUDA_Output[(j * w) + i] = color;
}

//Given a ray and a scene, find the closest hiting point
__host__ __device__ int intersectScene(Sphere *spheres, Plane *planes, Ray *ray, HitRecord *hit, int spherecount, int planecount, float tmin, float tmax){
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
	return hitSomething;
}

//Get the shading color at a hitting point
//Recursively calls itself on reflective and refractive surfaces
__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int lightcount){
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
		if(intersectScene(d_spheres, d_planes, &tempRay, &shadowed, spherecount, planecount, 0.01, sqrtf((lightDir.x * lightDir.x) + (lightDir.y * lightDir.y) + (lightDir.z * lightDir.z))) == 0){//No objects blocking the ray, do light calculation
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
	c->r += hit->material.Ie.g;
	c->r += hit->material.Ie.b;
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
	//Normalize(lightDir);
}

//Refract around a given normal and index of refraction
//Dir is assumed to be pointing into hit point
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *out){
	float mu;
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
	
	*out = *normal;
	
	if(cos_thetai > 0){
		Scale(out, (-mu * cos_thetai) + cos_thetar);
		ScaleAdd(out, mu, dir, out);
	} else {
		Scale(out, (-mu * cos_thetai) - cos_thetar);
		ScaleAdd(out, mu, dir, out);
	}
	
	Normalize(out);
}

//Find a reflected ray given an incoming ray and a surface normal
//Assumes dir is pointing away from the hit point
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *out){
	*out = *normal;
	Scale(out, 2 * VectorDot(dir, normal));
	VectorSub(out, out, dir);
}

//Find the intersection of a ray and a triangle
__host__ __device__ int triangleIntersect(Triangle *triangle, TriMesh *trimesh, Ray *ray, HitRecord *hit, float tmin, float tmax){
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
	
	//hit->material = triangle->material;
	hit->material = trimesh->material;
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
	CrossProduct(&(triangle->n), &v1, &v2);
	Normalize(&(triangle->n));
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