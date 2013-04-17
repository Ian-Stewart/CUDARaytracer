//Ian Stewart & Alexander Newman
//CUDA/OpenGL ray tracer

//Include standard headers
#include <stdio.h>
#include <stdlib.h>

//Include X
#include <X11/X.h>
#include <X11/Xlib.h>

//Include GLEW
#include <GL/glew.h>

//Include GLFW
#include <GL/glfw.h>

//Include other OpenGL
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/glut.h>

//Include CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//Include project headers
#include "raytraceutils.h"
#include "raystructs.h"
#include "raytracer.h"

int mouse_old_x;//Old mouse position
int mouse_old_y;

const unsigned int win_height = 480;//Window dimensions
const unsigned int win_width = 640;

Display 		*dpy;
Window 			root;
GLint att[] = 		{GLX_RGBA, GLX_DEPTH_SIZE,24,GLX_DOUBLEBUFFER, None};
XVisualInfo		*vi;
Colormap 		cmap;
XSetWindowAttributes 	swa;
Window 			win;
GLXContext 		glc;
XWindowAttributes 	gwa;
XEvent 			xev;

//Defined below main
void setUpXScreen();
void DrawAQuad();

__global__ void test_vbo_kernel(Color3f *c){
	c->r = 0;
	c->g = 0;
	c->b = 0;
}	

void launch_raytrace_kernel(){

}

int main(int argc, char *argv[]){
	setUpXScreen();
	
	//glutDisplayFunc(dpy);
	
	cudaGLSetGLDevice(0);//Assuming device 0 is the CUDA device
	//See page 51 of the CUDA C programming guide...
	
	// Register rescource with CUDA
	// map and unmap as many times as you want with cudaGraphicsMapResources() and cudaGraphicsUnmapResources()
	// cudaGraphicsResourceSetmapFlags() can be used to specify write-only usage hints for optimizations
	// Can map an OpenGL texture to CUDA using cudaGraphicsGLRegisterBuffer(). In CUDA, it appears as a device pointer and can be read and written by kernels or via cudaMemcpy() calls
	// 
	
	
	while(1){
	XNextEvent(dpy, &xev);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if(xev.type == Expose) {
		XGetWindowAttributes(dpy, win, &gwa);
		glViewport(0, 0, gwa.width, gwa.height);
		glXSwapBuffers(dpy, win);
	
		
	} else if(xev.type == KeyPress) {
		glXMakeCurrent(dpy, None, NULL);
		glXDestroyContext(dpy, glc);
		XDestroyWindow(dpy, win);
		XCloseDisplay(dpy);
		exit(0);
	}
	}//End while(1)
}

//Sets up the X screen so the main drawing loop can run
void setUpXScreen(){
	//Attempt to get a new display
	dpy = XOpenDisplay(NULL);
	if(dpy == NULL){
		printf("\n Cannot connect to X server\n");
		exit(0);
	}
	
	root = DefaultRootWindow(dpy);
	vi = glXChooseVisual(dpy, 0,att);
	if(vi == NULL){
		printf("\n No appropriate visual found\n");
		exit(0);
	} else {
		printf("\n Visual %p selected \n", (void *)vi->visualid);
	}
	
	cmap = XCreateColormap(dpy,root, vi->visual, AllocNone);
	
	swa.colormap = cmap;
	swa.event_mask = ExposureMask | KeyPressMask;
	
	win = XCreateWindow(dpy, root, 0, 0, win_width, win_height, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
	
	XMapWindow(dpy, win);
	XStoreName(dpy, win, "CUDA Ray Tracer - Ian Stewart & Alexander Newman");
	
	glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
	glXMakeCurrent(dpy, win, glc);
	
	//glEnable(GL_DEPTH_TEST);//May not need this for ray tracer	
}//End setUpXScreen