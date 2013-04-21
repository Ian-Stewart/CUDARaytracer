//This file contains structs needed for raytracer

//Represents a color as three floats (red, green, blue)
typedef struct {
	float r,g,b;
} Color3f;

//Represents a vector in 3D space
typedef struct {
	float x,y,z;
} Vector3f;

//A material. Contains color and light properties (reflective, transmissive, luminant)
//Some of these values could probably be removed to speed up rendering
typedef struct {
	Color3f Ka;	//Ambient reflectance
	Color3f Kd;	//Diffuse reflectance
	Color3f Ks;	//Specular reflectance
	Color3f Kr;	//Reflective color
	Color3f Kt;	//Transmitive (refractive) color
	Color3f Ie;	//Emissive color
	
	float phong_exp;//Used to calculate specular (phong specular exponent)
	float ior;	//Index of refraction
} Material;

//A triangle defined by three points and a normal
typedef struct {
	//Three points of triangle
	Vector3f p0,p1,p2;
	//Normal of triangle
	Vector3f n;
} Triangle;

//A sphere defined by center and radius
typedef struct {
	//Center of sphere
	Vector3f center;
	//Radius
	float radius;	
	//Material
	Material material;
} Sphere;

//Represents an array of triangles.
typedef struct {
	size_t length;
	Triangle *data;
} TriArray;

//A triangle mesh
//These are generated from .obj files
typedef struct {
	Material material;
	TriArray array;
} TriMesh;

//A plane
//Defines an infinite plane
typedef struct {
	Vector3f P0;
	Vector3f N;
} Plane;

//Hitrecord. Used upon intersection with an object in the scene
typedef struct {
	Material material;
	Vector3f normal;
	Vector3f pos;
	float t;
} HitRecord;

typedef struct{
	Vector3f pos;
	Color3f intensity;
} PointLight;