void initCamera(Camera *camera, Vector3f *in_eye, Vector3f *in_up, Vector3f *in_at, float in_fovy, float ratio);

void VectorNegate(Vector3f *v);

void VectorScale(Vector3f *v, float s);

void VectorNormalize(Vector3f *v);

void VectorScaleAdd(Vector3f *v0, Vector3f *v1, Vector3f *v2, float s);

void VectorSub(Vector3f *v0, Vector3f *v1, Vector3f *v2);

void initVector(Vector3f *v, float ix, float iy, float iz);

void VectorCrossProduct(Vector3f *out, Vector3f *v1, Vector3f *v2);