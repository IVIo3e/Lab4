#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_gl.h>
#include <GL/freeglut.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>   
#include <helper_cuda.h>
#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD         0.10f
#define REFRESH_DELAY     10 


const unsigned int window_width = 1024;
const unsigned int window_height = 720;

const unsigned int mesh_width = 512;
const unsigned int mesh_height = 512;

GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

float g_fAnim = 0.0;

float c_anim = 0.0;
float c_increase = 0.01;
int count = 0;


int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface* timer = NULL;


int fpsCount = 0;
int fpsLimit = 1;
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int* pArgc = NULL;
char** pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

bool runTest(int argc, char** argv, char* ref_file);
void cleanup();


bool initGL(int* argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

void display();
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void runCuda(struct cudaGraphicsResource** vbo_resource);
void runAutoTest(int devID, char** argv, char* ref_file);

const char* sSDKsample = "simpleGL (VBO)";

__global__ void simple_vbo_kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float mark = 0.027f;

    float u = (5 + cosf(x / 2 * mark) * sinf(y * mark) - sinf(x / 2 * mark) * sinf(2 * y * mark)) * cosf(x * mark);
    float v = (5 + cosf(x / 2 * mark) * sinf(y * mark) + sinf(x / 2 * mark) * sinf(2 * y * mark)) * sinf(x * mark) - 6;


    float w = 2.9 * sinf(x / 2 * mark + time) * sinf(y * mark) + 1.8 * cosf(x / 2 * mark + time) * sinf(2 * y * mark) - 3;

    pos[y * width + x] = make_float4(u, w, v, 1.0f);
}


void launch_kernel(float4* pos, unsigned int mesh_width,
    unsigned int mesh_height, float time)
{
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
}

bool checkHW(char* name, const char* gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main(int argc, char** argv)
{
    char* ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char**)argv, "file", (char**)&ref_file);
        }
    }

    printf("\n");

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == 60)
    {
        avgFPS = (60.f / (sdkGetTimerValue(&timer) / 1000.f)) * ((window_height * window_width * 1.0f) / (mesh_height * mesh_width * 1.0f));
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);
        printf("%.1f fps\n", avgFPS);
        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %5.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    glClearColor(0.0, 0.0, 0.1, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 100.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

bool runTest(int argc, char** argv, char* ref_file)
{
    sdkCreateTimer(&timer);

    int devID = findCudaDevice(argc, (const char**)argv);

    if (ref_file != NULL)
    {
        checkCudaErrors(cudaMalloc((void**)&d_vbo_buffer, mesh_width * mesh_height * 4 * sizeof(float)));

        runAutoTest(devID, argv, ref_file);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        glutDisplayFunc(display);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutCloseFunc(cleanup);

        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        runCuda(&cuda_vbo_resource);

        glutMainLoop();
    }

    return true;
}

void runCuda(struct cudaGraphicsResource** vbo_resource)
{
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
        *vbo_resource));

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void sdkDumpBin2(void* data, unsigned int bytes, const char* filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE* fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

void runAutoTest(int devID, char** argv, char* ref_file)
{
    char* reference_file = NULL;
    void* imageData = malloc(8 * mesh_width * mesh_height * sizeof(float));

    launch_kernel((float4*)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, 8 * mesh_width * mesh_height * sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, 8 * mesh_width * mesh_height * sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
            mesh_width * mesh_height * sizeof(float),
            MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags)
{
    assert(vbo);

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void display()
{
    sdkStartTimer(&timer);

    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0 + c_anim * 0.1, 0.0 + c_anim * 0.2, 0.0 + c_anim * 0.04 * g_fAnim);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    if (count++ > 1000) {
        count = 0;
        c_increase *= -1;
    }

    c_anim += c_increase;
    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}