#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

// Structure to represent a body
typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

// CUDA kernel for calculating gravitational forces using a grid-stride loop
__global__ void bodyForce(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        float px = p[i].x, py = p[i].y, pz = p[i].z;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - px;
            float dy = p[j].y - py;
            float dz = p[j].z - pz;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

int main(int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char* initialized_values;
    const char* solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else { 
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(Body);
    float* buf = (float*)malloc(bytes);
    Body* p = (Body*)buf;

    read_values_from_file(initialized_values, buf, bytes);

    double totalTime = 0.0;

    // Allocate GPU memory
    Body* d_p;
    cudaMalloc((void**)&d_p, bytes);

    // Get CUDA device properties to determine the number of SMs
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assumes GPU device 0

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // Copy data from CPU to GPU
        cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerSM = prop.maxThreadsPerMultiProcessor / threadsPerBlock;
        int numberOfBlocks = prop.multiProcessorCount * blocksPerSM;

        // Launch the bodyForce kernel with a grid-stride loop
        bodyForce<<<numberOfBlocks, threadsPerBlock>>>(d_p, dt, nBodies);

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // Copy data from GPU to CPU
        cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);

        // Integrate positions on CPU
        for (int i = 0; i < nBodies; i++) {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, buf, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Free GPU memory
    cudaFree(d_p);
    free(buf);

    return 0;
}
