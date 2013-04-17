//Ian Stewart & Alexander Newman
//CUDA/OpenGL ray tracer

//Include standard headers
#include <stdio.h>
#include <stdlib.h>

//Include X
#include <X11/X.h>
#include <X11/Xlib.h>

//Include GLEW. Apparently this is supposed to come before gl.h and glfw.h
#include <GL/glew.h>

//Include GLFW
#include <GL/glfw.h>

//Include other OpenGL
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>

//Include project headers
#include "raytraceutils.h"
#include "raystructs.h"
#include "raytracer.h"

int mouse_old_x;//Old mouse position
int mouse_old_y;

const unsigned int win_height = 640;//Window dimensions
const unsigned int win_width = 480;

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

int main(int argc, char *argv[]){
	setUpXScreen();
	
	while(1){
	XNextEvent(dpy, &xev);
	if(xev.type == Expose) {
		XGetWindowAttributes(dpy, win, &gwa);
		glViewport(0, 0, gwa.width, gwa.height);
		DrawAQuad(); 
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
	
	win = XCreateWindow(dpy, root, 0, 0, win_height, win_width, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
	
	XMapWindow(dpy, win);
	XStoreName(dpy, win, "CUDA Ray Tracer - Ian Stewart & Alexander Newman");
	
	glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
	glXMakeCurrent(dpy, win, glc);
	
	glEnable(GL_DEPTH_TEST);//May not need this for ray tracer	
}//End setUpXScreen

//Just a temporary thing. Will be replaced with code to draw objects
//Drawing will be done by CUDA
void DrawAQuad(){
	glClearColor(1.0,1.0,1.0,1.0);
	 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1., 1., -1., 1., 1., 20.);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0., 0., 10., 0., 0., 0., 0., 1., 0.);

	glBegin(GL_QUADS);
		glColor3f(1., 0., 0.); glVertex3f(-.75, -.75, 0.);
		glColor3f(0., 1., 0.); glVertex3f( .75, -.75, 0.);
		glColor3f(0., 0., 1.); glVertex3f( .75,  .75, 0.);
		glColor3f(1., 1., 0.); glVertex3f(-.75,  .75, 0.);
	glEnd();
}