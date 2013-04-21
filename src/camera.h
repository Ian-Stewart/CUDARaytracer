//camera.h
//Contains struct representing a camera

typedef struct {
	Vector3f eye;		//Position of camera
	Vector3f up;		//Up vector
	Vector3f at;		//Target
	float fovy;		//Field of View Y
	float aspect_ratio;	//Aspect ratio of image
	Vector3f corner;
	Vector3f center;
	Vector3f across;
} Camera;