
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "App.h"

using namespace glm;

//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
cudaError_t addWithCudav(vec2* c, vec2* a, vec2* b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// positions, directions, velocities
__global__ void addKernelv(vec2* c, vec2* a, vec2* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    //c[i] = vec2(1,2);
    //c[i].x = a[i].x + b[i].x;
    //c[i].y = a[i].y + b[i].y;
}

int main() {
    const int arraySize = 5;
	vec2 a[arraySize];
    vec2 b[arraySize];
    vec2 c[arraySize];

	for (int i = 0; i < arraySize; i++) {
		a[i] = vec2(i, i);
		b[i] = vec2(i*10, i*10);
		c[i] = vec2(0, 0);
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCudav(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	for (int i = 0; i < arraySize; i++) {
		printf("c[%d]=(%f, %f)\n", i, c[i].x, c[i].y);
	}

	App app;

	bool initOK = app.init();
	if (!initOK)
		exit(EXIT_FAILURE);

	app.run();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCudav(vec2* c, vec2* a, vec2* b, unsigned int size) {
    vec2* dev_a = 0;
    vec2* dev_b = 0;
    vec2* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernelv<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(vec2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
