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
__global__ void raytrace(Color3f *d_CUDA_Output, int w, int h);//This actually does the raytracing

__host__ void get_camera_ray(Ray *ray, Camera *d_camera, int w, int h, int i, int j);
__host__ __device__ void VectorSub(Vector3f *v, Vector3f *v1, Vector3f *v2);
__host__ __device__ void ScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s);
__host__ __device__ void InitVector(Vector3f *v, float ix, float iy, float iz);
__host__ __device__ void Normalize(Vector3f *v);
__host__ __device__ void VectorScale(Vector3f *v, float s);
__host__ __device__ void Negate(Vector3f *v);
__host__ __device__ void CrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2);

//Defined below main
void DrawScreen(SDL_Surface *screen);
void setpixel(SDL_Surface *screen, int x, int y, Uint8 r, Uint8 g, Uint8 b);
void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio);
unsigned int floatToUint(float f);