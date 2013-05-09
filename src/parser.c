#include "raystructs.h"
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

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

//calculates the crossproduct of two vectors and outputs it through the first argument pointer
void crossProduct(Vector3f *crossVector, Vector3f *v1, Vector3f *v2){
	crossVector->x = (v1->y * v2->z) - (v1->z * v2->z);
	crossVector->y = -(v1->x * v2->z) - (v1->z * v2->x);
	crossVector->z = (v1->x * v2->y) - (v1->y * v2->x);
}

//calculates the subtraction vector of two vectors outputs through the first argument pointer
void subtract(Vector3f *subVector, Vector3f *v1, Vector3f *v2){
	subVector->x = v1->x - v2->x;
	subVector->y = v1->y - v2->y;
	subVector->z = v1->z - v2->z;
}

//caclulates the normal of a triangle of three points outputs through the first argument
void calcNormal(Vector3f *n, Vector3f *p1, Vector3f *p2, Vector3f *p3){
	Vector3f one;
	Vector3f two;
	subtract(&one, p1, p2);
	subtract(&two, p1, p3);
	crossProduct(n, &one, &two);

}

//takes a face value and outputs a triangle outputs through the first argument
void triangulate(Triangle *out, int *face){
	out->p0 = vertices[face[0]];
	out->p1 = vertices[face[1]];
	out->p2 = vertices[face[2]];
	calcNormal(&(out->n), &(out->p0), &(out->p1), &(out->p2));
}

//sets the size of the local int ** for faces
void setFacesSize(){
	faces = malloc(sizeof(int*) * numberOfFaces);
	for(int i = 0; i < numberOfFaces; i++){
		faces[i] = malloc( sizeof(int) * 3 );
	}
	faces[9][0] = 1;
}

//sets the size of the local Vector3f * for vertices
void setVerticesSize(){
	vertices = (Vector3f*)malloc( sizeof(Vector3f) * numberOfVertices);
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
	int currentVertex = 0;
	FILE *file = fopen(filename, "r");

	if(file != NULL){
		char line [128];
		char * placeholder;
		while(fgets (line, sizeof line, file) != NULL){
			if(line[0] == 'v'){
				//assign values to the array then increment
				//need to go over pointer logic here!!!!!!!!!!!!!!!!!!!!!!!!!!
				sscanf(line, "%*s %f %f %f", &(vertices[currentVertex].x), &(vertices[currentVertex].y), &(vertices[currentVertex].z));
				currentVertex++;
			}
			if(line[0] == 'f'){
				//assign values to the array then increment
				sscanf(line, "%*s %d %d %d", &(faces[currentFace][0]), &(faces[currentFace][1]), &(faces[currentFace][2]));
				//printf("%d %d %d\n", faces[currentFace][0], faces[currentFace][1], faces[currentFace][2]);
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

//creates a Vector3f from 3 floats, outputs from the first argument
void createVertex(Vector3f * vertex, float x, float y, float z){
	vertex->x = x;
	vertex->y = y;
	vertex->z = z;
}

//constructs two triangles to form the surface of a box, outputs fromt he first 2 arguments
void createSquareFace(Triangle * t1, Triangle * t2, Vector3f v1, Vector3f v2, Vector3f v3, Vector3f v4){
	//to make sure sure the box is formed correctly v1 and v3 must be opposite corners as do v2 and v4
	//this means between v1 and v3 x,y,z do not share any values besides the plane they are on
	t1->p0 = v1;
	t1->p1 = v2;
	t1->p2 = v4;
	//May need to correct pointer logic in this method!!!!!!!!!!!!!!!!!!!1
	calcNormal(&(t1->n), &v1, &v2, &v4);
	t2->p0 = v3;
	t2->p1 = v4;
	t2->p2 = v2;
	calcNormal(&(t1->n), &v3, &v4, &v2);
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
	Vector3f * bbv = malloc(sizeof(Vector3f) * 8);
	createVertex(&bbv[0], minX, minY, minZ);
	createVertex(&bbv[1], maxX, minY, minZ);
	createVertex(&bbv[2], minX, maxY, minZ);
	createVertex(&bbv[3], minX, minY, maxZ);
	createVertex(&bbv[4], maxX, minY, maxZ);
	createVertex(&bbv[5], maxX, maxY, minZ);
	createVertex(&bbv[6], minX, maxY, maxZ);
	createVertex(&bbv[7], maxX, maxY, maxZ);

	//next contruct the box using two triangles per face
	Triangle *bbf = malloc(sizeof(Triangle) * 12);
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
void createTriArrayFromFile(TriArray * out, TriArray * bb, char * filename){
	setFileSize(filename);
	//printf("faces %d\n", numberOfFaces);
	//printf("vertices %d\n", numberOfVertices);
	readInObject(filename);
	//printf("read in complete\n");
	Triangle *triangles = malloc(sizeof(Triangle) * numberOfFaces);
	if(triangles == NULL){printf("triangle malloc failed");}
	for(int i = 0; i < numberOfFaces; i++){
		//printf("before triangluate %d\n", i);
		triangulate(&(triangles[i]), faces[i]);
	}
	//printf("triangles created\n");
	out->length = numberOfFaces;
	out->data = triangles;
	//printf("length %d\n", numberOfFaces);
	
	createBoundingBox(bb);
	//printf("bounding box created\n");
	free(faces);
	free(vertices);
}

/*
for testing purposes
int main(){
	TriArray object;
	TriArray box;
	createTriArrayFromFile(&object, &box, "../Models/diamond.obj");
	int i = 0;
	for(i = 0; i < object.length; i++){
		printf("\nPoint 1 %3.3f ", object.data[i].p0.x);
		printf("%3.3f ", object.data[i].p0.y);
		printf("%3.3f\n", object.data[i].p0.z);
		printf("Point 2 %3.3f ", object.data[i].p0.x);
		printf("%3.3f ", object.data[i].p0.y);
		printf("%3.3f\n", object.data[i].p0.z);
		printf("Point 3 %3.3f ", object.data[i].p0.x);
		printf("%3.3f ", object.data[i].p0.y);
		printf("%3.3f\n", object.data[i].p0.z);
		printf("Normal %3.3f", object.data[i].n.x);
		printf("%3.3f ", object.data[i].n.y);
		printf("%3.3f\n", object.data[i].n.z);
	}
} */