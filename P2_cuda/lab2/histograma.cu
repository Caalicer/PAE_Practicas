#include <stdio.h>
#include <stdlib.h>
#include "pgmio.h"

#define GRAY_LEVELS 256

// Versión con memoria global
__global__ void histogramGlobal(const unsigned char* image, int* hist,
                                int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = width * height;

    if (idx < size) {
        atomicAdd(&hist[image[idx]], 1);
    }
}

// Versión con memoria compartida
__global__ void histogramShared(const unsigned char* image, int* hist,
                                int width, int height) {
    __shared__ int histoPrivate[GRAY_LEVELS];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = width * height;

    if (threadIdx.x < GRAY_LEVELS) {
        histoPrivate[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < size) {
        atomicAdd(&histoPrivate[image[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x < GRAY_LEVELS) {
        atomicAdd(&hist[threadIdx.x], histoPrivate[threadIdx.x]);
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {

        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width, height;
    unsigned char* image;

    image = loadPGM(argv[1], &width, &height);

    int *d_hist, *d_image;
    cudaMalloc(&d_image, width * height * sizeof(unsigned char));
    cudaMalloc(&d_hist, GRAY_LEVELS * sizeof(int));
    cudaMemset(d_hist, 0, GRAY_LEVELS * sizeof(int));

    cudaMemcpy(d_image, image, width * height * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    histogramGlobal<<<numBlocks, blockSize>>>(d_image, d_hist, width, height);

    int h_hist[GRAY_LEVELS];
    cudaMemcpy(h_hist, d_hist, GRAY_LEVELS * sizeof(int),
               cudaMemcpyDeviceToHost);

    printHistogram(h_hist);

    cudaFree(d_image);
    cudaFree(d_hist);
    free(image);

    return 0;
}
