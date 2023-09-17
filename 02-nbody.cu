#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

__global__ void bodyForceKernel(const Body* __restrict__ p, Body* __restrict__ newp, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        newp[i].vx += dt * Fx;
        newp[i].vy += dt * Fy;
        newp[i].vz += dt * Fz;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char* initialized_values;
    const char* solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float* buf;

    buf = (float*)malloc(bytes);

    Body* p = (Body*)buf;
    Body* newp = (Body*)malloc(bytes);

    read_values_from_file(initialized_values, buf, bytes);

    double totalTime = 0.0;

    // Allocate GPU memory for bodies
    Body* d_p;
    Body* d_newp;
    cudaMalloc((void**)&d_p, bytes);
    cudaMalloc((void**)&d_newp, bytes);

    // Transfer data from CPU to GPU
    cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

    // Specify thread block size and grid size for CUDA kernels
    int blockSize = 256;
    int numBlocks = (nBodies + blockSize - 1) / blockSize;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // Launch bodyForce kernel
        bodyForceKernel<<<numBlocks, blockSize>>>(d_p, d_newp, dt, nBodies);

        // Synchronize to ensure completion of bodyForce
        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;

        // Swap the pointers to update positions
        Body* temp = d_p;
        d_p = d_newp;
        d_newp = temp;
    }

    // Transfer data back from GPU to CPU
    cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_p);
    cudaFree(d_newp);

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, buf, bytes);

    // Print performance metric
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Free CPU memory
    free(buf);
    free(newp);

    return 0;
}
