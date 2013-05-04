//Ian Stewart & Alexander Newman
//CUDA/SDL ray tracer

#include <stdio.h>
#include <stdlib.h>
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

//__host__ __device__ indicates a function that is run on both the GPU and CPU
//__global__ indicates a CUDA kernel
__global__ void raytrace(Color3f *d_CUDA_Output, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Camera *d_camera, int spherecount, int planecount, int lightcount, int w, int h, int c);//This actually does the raytracing

__host__ __device__ int sphereIntersect(Sphere *sphere, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int planeIntersect(Plane *plane, Ray *ray, HitRecord *hit, float tmin, float tmax);
__host__ __device__ int intersectScene(Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int lightcount, float tmin, float tmax);
__host__ __device__ int triangleIntersect(Triangle *triangle, TriMesh *trimesh, Ray *ray, HitRecord *hit, float tmin, float tmax);

__host__ __device__ float VectorDot(Vector3f *v, Vector3f *u);
__host__ __device__ float findDeterminant(Vector3f *col0, Vector3f *col1, Vector3f *col2);

__host__ __device__ inline void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ inline void InitColor(Color3f *c, float ir, float ig, float ib);

__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int lightcount, int depth);
__host__ __device__ void getLight(PointLight *light, Vector3f *p, Vector3f *pos, Vector3f *lightDir, Color3f *c);
__host__ __device__ void getCameraRay(Ray *ray, Camera *d_camera, float x, float y);
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *refr);
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *refl);
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
	InitVector(&eye, -1, 0, 0);
	InitVector(&at, 0, 0, 0);
	InitVector(&up, 0,0,1);
	initCamera(&camera, &eye, &up, &at, 45, 1);//Set up camera

	//cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
	
	SDL_Surface *screen;
	SDL_Event event;
	

	int c = 0;//For basic animation
	int keypress = 0;
	//int totalTris = 0;
	//int meshcount = 0;
	int spherecount = 3;
	int planecount = 1;
	int lightcount = 1;
	
	Sphere *spheres 	= (Sphere *)	malloc(sizeof(Sphere) * 3);//Scene will have three spheres
	//TriMesh *meshes 	= (TriMesh *)	malloc(sizeof(TriMesh) * 3);//Three trimeshes
	Plane *planes 		= (Plane *)	malloc(sizeof(Plane) * 1);//One plane
	PointLight *lights 	= (PointLight *)malloc(sizeof(PointLight) * 1);//One light
	
	InitVector(&(spheres[0].center), 0, 0, 0);
	InitVector(&(spheres[1].center), 3, -1, 1);
	InitVector(&(spheres[2].center), -1, 1, -1);
	
	spheres[0].radius = 1;
	spheres[1].radius = 0.75;
	spheres[2].radius = 1.5;
	
	InitVector(&(planes[0].p), 0, 0, -4);
	InitVector(&(planes[0].normal), 0, 1, 1);
	Normalize(&(planes[0].normal));
	
	InitVector(&(lights[0].pos), 0, 4, 8);
	InitColor(&(lights[0].intensity), 100,100,100);
	
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
	
	/*
	for(i = 0; i < meshcount; i++){
		totalTris += meshes[i].triangles;//Count up the total number of triangles
	}
	totalTris += meshcount * 12;//For bounding triangles
	*/
	
	//cudaMalloc(&d_triangles, sizeof(Triangle) * totalTris);//Allocate space for flattened triangle data
	//h_flattened_triangles = (Triangle *)malloc(sizeof(Triangle) * totalTris);
	//Now dump triangle data to flat array.
	//Will be accessed by pointer shenanegans
	//Running 'offset' based on number of triangles in previous trimeshes
	//In this model, bounding volumes are the first 12 triangles of each trimesh segment
	//int offset = 0;
	//Triangle* currentptr;//Used in copying 
	//currentptr = h_flattened_triangles;
	//for(int i = 0; i < meshes; i++){
	//	memcpy(h_flattened_triangles, 
	//	currentptr += 12;//Increment pointer after copying bounding triangles
	//}
	
	//Finally, copy flattened data
	//cudaMemcpy(d_triangles, h_flattened_triangles, sizeof(Triangle) * totalTris, cudaMemcpyHostToDevice);

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
	while(!keypress){
		//Launch Kernel
		raytrace<<<numBlocks, threadsPerBlock>>>(
			(Color3f *)d_CUDA_Output,
			(Sphere *) d_spheres,
			(Plane *) d_planes,
			(PointLight *) d_lights,
			(Camera *)d_camera,
			spherecount,
			planecount,
			lightcount,
			WIDTH,
			HEIGHT,
			c++
		);
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
//Size of each array of objects is given by 'x'count integers
__global__ void raytrace(
				Color3f *d_CUDA_Output,
				Sphere *d_spheres,
				Plane *d_planes,
				PointLight *d_lights,
				Camera *d_camera,
				int spherecount,
				int planecount,
				int lightcount,
				int w,
				int h,
				int c
			){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	Ray cameraRay;
	InitVector(&(cameraRay.d), 1, 0, 0);
	float x;
	float y; //(x,y) is the coordinate for this point in the image such that 0<=x,y<=1
	
	x = (float) i/ (float) w;
	y = (float) j/ (float) h;
	getCameraRay(&cameraRay, d_camera, x, y);//Get the camera ray
	
	float tmin = 0.001;
	float tmax = 1000;
	HitRecord hit;

	d_CUDA_Output[(j * w) + i].r = 0;
	d_CUDA_Output[(j * w) + i].g = 0;
	d_CUDA_Output[(j * w) + i].b = 0;
	
	if(intersectScene(d_spheres, d_planes, d_lights, &cameraRay, &hit, spherecount, planecount, lightcount, tmin, tmax) == 1){//Ray hit something in the scene
		Color3f c;
		getShadingColor(&c, d_spheres, d_planes, d_lights, &cameraRay, &hit, spherecount, planecount, lightcount, 0);
		d_CUDA_Output[(j * w) + i] = c;
	}
}

//Given a ray and a scene, find the closest hiting point
__host__ __device__ int intersectScene(
					Sphere *d_spheres,
					Plane *d_planes,
					PointLight *d_lights,
					Ray *ray,
					HitRecord *hit,
					int spherecount,
					int planecount,
					int lightcount,
					float tmin,
					float tmax
				){
	int hitSomething = 0;//If ray intersects with no objects, return zero. Otherwise return 1.
	int i;//,j;
	
	//Check spheres
	for(i = 0; i < spherecount; i++){
		if(sphereIntersect(&(d_spheres[i]), ray, hit, tmin, tmax)  == 1){//ray intersects with sphere //TODO change &(spheres[i]) to spheres + i?
			hitSomething = 1;
			tmax = hit->t;//Reduce range, all hits after this must be closer to ray origin
		}
	}
	
	//Check triangle meshes
	//TODO: Check bounding volumes first to avoid checking every triangle needlessly
	/*
	for(i = 0; i < meshcount; i++){
		for(j = 0; j < meshes[i].triangles; j++){//Go through every triangle in the mesh
			if(triangleIntersect(&(meshes[i].data[j]), ray, hit, tmin, tmax) == 1){//ray intersects triangle
				hitSomething = 1;
				tmax = hit->t;
			}//end if
		}//end for(j = 0; j < scene->meshes[j].triangles...)
	}//End for(i = 0; i < scene->meshcount...)
	*/
	//Check planes
	for(i = 0; i < planecount; i++){//Check to see if ray intersects any planes
		if(planeIntersect(&(d_planes[i]), ray, hit, tmin, tmax) == 1){//ray intersects with plane
			hitSomething = 1;
			tmax = hit->t;
		}//endif
	}//end for (i = 0; i < scene->planecount...)
	
	return hitSomething;
}

//Get the shading color at a hitting point
//Recursively calls itself on reflective and refractive surfaces
__host__ __device__ void getShadingColor(Color3f *c, Sphere *d_spheres, Plane *d_planes, PointLight *d_lights, Ray *ray, HitRecord *hit, int spherecount, int planecount, int lightcount, int depth){
	InitColor(c, 0, 0, 0);
	Color3f lightColor;
	Vector3f lightPos, lightDir, flippedRay, R;
	Ray lightRay;
	HitRecord tempHit;
	float lightDist;
	int i;
	//Iterate through lights to find surface shading color
	for(i = 0; i < lightcount; i++){
		getLight(&(d_lights[i]), &(hit->pos), &lightPos, &lightDir, &lightColor);
		
		//Now check if shadowed
		lightDist = sqrtf((lightDir.x * lightDir.x) + (lightDir.y * lightDir.y) + (lightDir.z * lightDir.z));
		Normalize(&lightDir);
		
		lightRay.d = lightDir;
		lightRay.o = hit->pos;
		if(intersectScene(d_spheres, d_planes, d_lights, &lightRay, &tempHit, spherecount, planecount, lightcount, 0.01, lightDist) == 0){//No objects blocking the ray, do light calculation
			//Add diffuse portion
			c->r += lightColor.r * hit->material.Kd.r * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			c->g += lightColor.g * hit->material.Kd.g * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			c->b += lightColor.b * hit->material.Kd.b * fmaxf(VectorDot(&(hit->normal), &lightDir), 0);
			
			//Add specular portion
			//lightDir is the normalized vector from hit to light
			//lightPos is the position of lightColor
			//lightRay is the ray from hit to light
			
			Reflect(&(lightRay.d), &(hit->normal), &R);
			
			flippedRay = ray->d;
			Negate(&flippedRay);
			
			c->r += lightColor.r * hit->material.Ks.r * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
			c->g += lightColor.g * hit->material.Ks.g * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
			c->b += lightColor.b * hit->material.Ks.b * pow(fmaxf(0,VectorDot(&R, &flippedRay)),hit->material.phong_exp);
		}//end if(intersectScene() == 0)
	}//End light shading loop

	if(depth < MAX_DEPTH){
		Color3f reflectedColor, refractedColor;
		InitColor(&reflectedColor, 0, 0, 0);
		InitColor(&refractedColor, 0, 0, 0);
		Ray reflectedRay, refractedRay;
		HitRecord refractHit, reflectHit;
		
		//intersectScene(Scene *scene, Ray *ray, HitRecord *hit, float tmin, float tmax)
		
		//Find reflective portion
		if(hit->material.Kr.r > 0 || hit->material.Kr.g > 0 || hit->material.Kr.b > 0){//Surface is reflective
			reflectedRay.o = hit->pos;
			reflectedRay.d = ray->d;
			Negate(&(reflectedRay.d));
			Reflect(&(reflectedRay.d), &(hit->normal), &(reflectedRay.d));
				if(intersectScene(d_spheres, d_planes, d_lights, &reflectedRay, &reflectHit, spherecount, planecount,lightcount, 0.01, 1000) == 1){//reflected ray hits something
				getShadingColor(
					&reflectedColor, 
					d_spheres,
					d_planes,
					d_lights,
					&reflectedRay,
					&reflectHit,
					spherecount,
					planecount,
					lightcount,
					depth + 1
				);//Recursive call to get reflected color
					c->r += reflectHit.material.Kr.r * reflectedColor.r;
					c->g += reflectHit.material.Kr.g * reflectedColor.g;
					c->b += reflectHit.material.Kr.b * reflectedColor.b;
				}
		}//End if reflective
		
		//Find refractive portion
		if(hit->material.Kr.r > 0 || hit->material.Kr.r > 0 || hit->material.Kr.r > 0){//Material has refractive properties
			refractedRay.o = hit->pos;
			refractedRay.d = ray->d;
			Refract(&(ray->d), &(hit->normal), hit->material.ior, &(refractedRay.d));
			
			if(intersectScene(d_spheres, d_planes, d_lights, &refractedRay, &refractHit, spherecount, planecount,lightcount, 0.01, 1000) == 1){
				getShadingColor(
					&refractedColor, 
					d_spheres,
					d_planes,
					d_lights,
					&refractedRay,
					&refractHit,
					spherecount,
					planecount,
					lightcount,
					depth + 1
				);//Recursive call
				
				//Hack - refracted shading color is made more or less strong depending on the length of the ray through the objects
				//Seems to make stuff look really nice
				Vector3f vThroughObj = refractHit.pos;
				VectorSub(&vThroughObj, &vThroughObj, &(hit->pos));
				float factor = 1/(sqrtf((vThroughObj.x * vThroughObj.x) + (vThroughObj.y * vThroughObj.y) + (vThroughObj.z * vThroughObj.z)));
				if(factor > 1){
					c->r += hit->material.Kt.r * refractedColor.r;
					c->g += hit->material.Kt.g * refractedColor.g;
					c->b += hit->material.Kt.b * refractedColor.b;
				} else {
					c->r += hit->material.Kt.r * refractedColor.r * factor;
					c->g += hit->material.Kt.g * refractedColor.g * factor;
					c->b += hit->material.Kt.b * refractedColor.b * factor;
				}
			}//End if(intersectScene())
		}//End refractive section
		
		//Add in emissive portion of material
		c->r += hit->material.Ie.r;
		c->g += hit->material.Ie.g;
		c->b += hit->material.Ie.b;
		
		//Add in ambient light portion
		//Not currently implemented
	}
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
__host__ __device__ void Refract(Vector3f *dir, Vector3f *normal, float ior, Vector3f *refr){
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
	
	Vector3f out = *normal;
	
	if(cos_thetai > 0){
		Scale(&out, (-mu * cos_thetai) + cos_thetar);
		ScaleAdd(&out, mu, dir, &out);
	} else {
		Scale(&out, (-mu * cos_thetai) + cos_thetar);
		ScaleAdd(&out, mu, dir, &out);
	}
	
	Normalize(&out);
	*refr = out;
}

//Find a reflected ray given an incoming ray and a surface normal
//Assumes dir is pointing away from the hit point
__host__ __device__ void Reflect(Vector3f *dir, Vector3f *normal, Vector3f *refl){
	*refl = *normal;
	Scale(&(*refl), 2 * VectorDot(dir, normal));
	VectorSub(refl, refl, dir);
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
	return (v->x * u->x) + (v->y * u->y) + (v->z * u->z);
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