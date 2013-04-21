//raytraceutils.c
//Contains lots of functions related to structs defined in raystructs.h
#include <Math.h>

#ifndef PI
#define PI           3.14159265358979323846
#endif


//initCamera
//Run to find useful values for camera operations later.
//Will need to be run every time the camera moves
void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio){
	//Update camera information
	camera->eye = *in_eye;
	camera->up = *in_up;
	camera->at = *in_at;
	camera->fovy = in_fovy;
	camera->aspect_ratio = ratio;
	
	//Compute points of image plane
	float dist = 1;
	float top = dist * (tanf((camera->fovy * PI)/360);
	float bottom = -top;
	float right = ratio*top;
	float left = right;
	
	Vector3f gaze;
	initVector(&gaze, 0, 0, 0);
	VectorSub(
}

//sets v = v/|v|
void VectorNormalize(Vector3f *v){
	float magnitude = sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));//Length of vector v
	v->x = (v->x)/magnitude;
	v->y = (v->y)/magnitude;
	v->z = (v->z)/magnitude;
}

//Vector operation v0 = v1-v2
void VectorSub(Vector3f *v0, Vector3f *v1, Vector3f *v2){
	v0->x = (v1->x) - (v2->x);
	v0->y = (v1->y) - (v2->y);
	v0->z = (v1->z) - (v2->z);
}

//Initializes vector
void initVector(Vector3f *v, float ix, float iy, float iz){
	v->x = ix;
	v->y = iy;
	v->z = iz;
}