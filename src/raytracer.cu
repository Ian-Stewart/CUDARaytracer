//Ian Stewart & Alexander Newman
//CUDA/OpenGL ray tracer

//Include standard headers
#include <stdio.h>
#include <stdlib.h>

//Include GLEW. Apparently this is supposed to come before gl.h and glfw.h
#include <GL/glew.h>

//Include GLFW
#include <GL/glfw.h>

//Include project headers
//#include <./raytraceutils.h>
//#include <./raystructs.h>

int mouse_old_x, mouse_old_y;//Old mouse position

const unsigned int win_height = 640;//Window dimensions
const unsigned int win_width = 480;

int main(){
	if(!glfwInit()) {
		fprintf(stderr, "Failed to intialize GLFW\n");
		return -1;
	}

	glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4); // 4x antialiasing
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
	
	// Open a window and create its OpenGL context
	if( !glfwOpenWindow( win_width, win_height, 0,0,0,0, 32,0, GLFW_WINDOW ) ) {
		fprintf( stderr, "Failed to open GLFW window\n" );
		glfwTerminate();
		return -1;
	}
	
	// Initialize GLEW
	glewExperimental=true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	glfwSetWindowTitle( "Tutorial 01" );
	
	//Ensure we can capture the excape key being pressed
	glfewEnable(GLFW_STICKY_KEYS);
	
	do{
		//Draw nothing
		glfwSwapBuffers();
	} while (glfwGetKey(GLFW_KEY_ESC) != GLFW_PRESS && glfwGetWindowParam(GLFW_OPENED);

}