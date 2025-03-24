#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "pgmio.h"

#define GRAY_LEVELS 256

double get_time() {

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
}

void checkCUDAError(const char* msg) {

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {

        printf("Cuda error: %s: %s in %s at line %d.\n", msg,
               cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void cpu_histogram(unsigned char* input, int* histogram, size_t imageSize) {

    for (size_t i = 0; i < imageSize; i++) {

        histogram[input[i]]++;
    }
}

__global__ void histogram(unsigned char* input, int* globalHistogram,
                          size_t imageSize) {

    __shared__ unsigned int sharedHistogram[GRAY_LEVELS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int localThreadID = threadIdx.x;

    for (int i = localThreadID; i < GRAY_LEVELS; i += blockDim.x) {

        sharedHistogram[i] = 0;
    }

    __syncthreads();

    for (int i = tid; i < imageSize; i += stride) {

        unsigned char pixelValue = input[i];
        atomicAdd(&sharedHistogram[pixelValue], 1);
    }

    __syncthreads();

    for (int i = localThreadID; i < GRAY_LEVELS; i += blockDim.x) {

        if (sharedHistogram[i] > 0) {

            atomicAdd(&globalHistogram[i], sharedHistogram[i]);
        }
    }
}

int main(int argc, char** argv) {

    if (argc < 3) {

        printf("Usage: %s <image.pgm> <threadsPerBlock>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* imagePath = argv[1];
    int threadsPerBlock = atoi(argv[2]);

    double startAlloc, startAlloc2, endAlloc;
    double startInit, endInit;
    double startHistogram, endHistogram;
    double startHD, endHD;
    double startDH, endDH;
    double overhead;

    int w;
    int h;

    unsigned char* h_image = loadPGMu8(imagePath, &w, &h);

    if (h_image == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    size_t imageSize = w * h;
    unsigned char* d_image;
    int* d_hist;
    int h_hist[GRAY_LEVELS] = {0};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("Getting device properties");

    startAlloc = get_time();
    startAlloc2 = get_time();

    cudaMalloc(&d_image, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_hist, GRAY_LEVELS * sizeof(int));

    checkCUDAError("Allocating device memory for image and histogram");

    endAlloc = get_time();

    startHD = get_time();

    cudaMemcpy(d_image, h_image, imageSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    endHD = get_time();

    checkCUDAError("Copying image data to device");

    startInit = get_time();

    cudaMemset(d_hist, 0, GRAY_LEVELS * sizeof(int));

    endInit = get_time();

    checkCUDAError("Initializing histogram array");

    long blocksPerGrid = ceil(imageSize / threadsPerBlock);

    if (blocksPerGrid > prop.maxGridSize[0]) {

        blocksPerGrid = prop.maxGridSize[0];
    }

    startHistogram = get_time();

    histogram<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_hist, imageSize);

    cudaDeviceSynchronize();

    endHistogram = get_time();

    checkCUDAError("Executing shared memory histogram kernel");

    startDH = get_time();

    cudaMemcpy(h_hist, d_hist, GRAY_LEVELS * sizeof(int),
               cudaMemcpyDeviceToHost);

    endDH = get_time();

    checkCUDAError("Copying histogram results to host");

    int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, histogram,
                                                  threadsPerBlock, 0);
    checkCUDAError("Calculating occupancy");

    float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) /
                      prop.maxThreadsPerMultiProcessor;

    overhead = startAlloc2 - startAlloc;
    double alloc_time = endAlloc - startAlloc2 - overhead;
    double init_time = endInit - startInit - overhead;
    double histogram_time = endHistogram - startHistogram - overhead;
    double hd_time = endHD - startHD - overhead;
    double dh_time = endDH - startDH - overhead;
    double total_time =
        alloc_time + init_time + histogram_time + hd_time + dh_time;

    printf("\nPAE,%s,%d,%ld,%d,%f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%"
           "ld,PAE\n",
           imagePath, threadsPerBlock, blocksPerGrid, maxBlocksPerSM, occupancy,
           overhead, alloc_time, init_time, histogram_time, hd_time, dh_time,
           total_time, imageSize);

    int h_hist_cpu[GRAY_LEVELS] = {0};

    cpu_histogram(h_image, h_hist_cpu, imageSize);

    int errors = 0;

    for (int i = 0; i < GRAY_LEVELS; i++) {

        if (h_hist[i] != h_hist_cpu[i]) {

            errors = 1;
            break;
        }
    }

    if (errors == 0) {

        printf("Verification PASSED! CPU and GPU results match.\n");

    } else {

        printf("Verification FAILED! CPU and GPU results do not match.\n");
    }

    cudaFree(d_image);
    checkCUDAError("Freeing device memory for image");

    cudaFree(d_hist);
    checkCUDAError("Freeing device memory for histogram");

    free(h_image);

    return EXIT_SUCCESS;
}