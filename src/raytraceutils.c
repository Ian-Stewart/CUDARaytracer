//raytraceutils.c
//Contains lots of functions related to structs defined in raystructs.h
#include <Math.h>

//Include structs and such
#include "camera.h"
#include "raystructs.h"

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
	VectorSub(&gaze, &(camera->at), &(camera->eye));//gaze = at-eye
	
	camera->center = camera->eye;
	Vector3f W = gaze;
	VectorNegate(&W);
	VectorNormalize(&W);
	Vector3f V = camera->up;
	Vector3f U;
	initVector(&U, 0, 0, 0);
	VectorCrossProduct(&U, &V, &W);//U = VxW
	VectorNormalize(&U);
	VectorCrossProduct(&V, &W, &U);
	
	initVector(&(camera->corner));
	VectorScaleAdd(&(camera->corner), &U, &(camera->center), left);
	VectorScaleAdd(&(camera->corner), &V, &(camera->corner), bottom);
	VectorScaleAdd(&(camera->corner), &W, &(camera->corner), -dist);
	
	initVector(&(camera->across),U.x, U.y, U.z);
	VectorScale(&(camera->across), right-left);
	
	initVector(&(camera->up), V.x, V.y, V.z);
	VectorScale(&(camera->up), top-bottom);
}

//Negates a vector
void VectorNegate(Vector3f *v){
	v->x = -(v->x);
	v->y = -(v->y);
	v->z = -(v->z);
}

//Scales a vector v = s*v
void VectorScale(Vector3f *v, float s){
	v->x = s*(v->x);
	v->y = s*(v->y);
	v->z = s*(v->z);
}

//sets v = v/|v|
void VectorNormalize(Vector3f *v){
	float magnitude = sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));//Length of vector v
	v->x = (v->x)/magnitude;
	v->y = (v->y)/magnitude;
	v->z = (v->z)/magnitude;
}

//sets v0 = (s*v1 + v2)
void VectorScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s){
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

//Compute the cross product of a vector
//v1 x v2 = |{{i,j,k},{v1.x,v1.y,v1.z},{v2.x,v2.y,v2.z}}|
void VectorCrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2){
	out->x =  (v1->y * v2->z) - (v1->z * v2->z);
	out->y = -(v1->x * v2->z) - (v1->z * v2->x);
	out->z =  (v1->x * v2->y) - (v1->y * v2->x);
}