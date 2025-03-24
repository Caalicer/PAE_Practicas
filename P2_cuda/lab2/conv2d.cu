#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "pgmio.h"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
}

int conv2d_sequential(float* image, int img_w, int img_h, float* conv) {
    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            float sum = 0.0;
            for (int ky = 0; ky < ker_h; ky++) {
                for (int kx = 0; kx < ker_w; kx++) {
                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;
                    if (img_x >= 0 && img_x < img_w && img_y >= 0 &&
                        img_y < img_h) {
                        sum +=
                            image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                    }
                }
            }
            conv[y * img_w + x] = sum;
        }
    }
    return EXIT_SUCCESS;
}

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s in %s at line %d.\n", msg,
                cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void conv2dKernel(float* image, float* conv, int img_w, int img_h,
                             float* ker, int ker_w, int ker_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_w && y < img_h) {
        int pad_w = ker_w / 2;
        int pad_h = ker_h / 2;
        float sum = 0.0f;

        for (int ky = 0; ky < ker_h; ky++) {
            for (int kx = 0; kx < ker_w; kx++) {
                int img_x = x + kx - pad_w;
                int img_y = y + ky - pad_h;
                if (img_x >= 0 && img_x < img_w && img_y >= 0 &&
                    img_y < img_h) {
                    sum += image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                }
            }
        }

        conv[y * img_w + x] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.pgm> <threadsPerBlock>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* imagePath = argv[1];
    int threadsPerBlock = atoi(argv[2]);

    double startAlloc, startAlloc2, endAlloc;
    double startInit, endInit;
    double startConv, endConv;
    double startHD, endHD;
    double startDH, endDH;
    double overhead;

    int img_w; // Ancho
    int img_h; // Alto

    float* h_image = loadPGM32(imagePath, &img_w, &img_h);

    if (h_image == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    size_t imageSize = img_w * (size_t)img_h;
    size_t imageSizeBytes = imageSize * sizeof(float);

    int ker_w = 3, ker_h = 3;
    float h_ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    size_t kernelSizeBytes = ker_w * ker_h * sizeof(float);
    float *d_image, *d_conv, *d_ker;
    float* h_conv = NULL;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("Getting device properties");

    startAlloc = get_time();
    startAlloc2 = get_time();

    cudaMalloc(&d_image, imageSizeBytes);
    checkCUDAError("Allocating device memory for image");

    cudaMalloc(&d_conv, imageSizeBytes);
    checkCUDAError("Allocating device memory for convolution result");

    cudaMalloc(&d_ker, kernelSizeBytes);
    checkCUDAError("Allocating device memory for kernel");

    h_conv = (float*)malloc(imageSizeBytes);

    if (h_conv == NULL) {
        printf("Error allocating host memory for convolution result\n");
        return EXIT_FAILURE;
    }

    endAlloc = get_time();

    startHD = get_time();

    cudaMemcpy(d_image, h_image, imageSizeBytes, cudaMemcpyHostToDevice);
    checkCUDAError("Copying image data to device");

    cudaMemcpy(d_ker, h_ker, kernelSizeBytes, cudaMemcpyHostToDevice);
    checkCUDAError("Copying kernel data to device");

    endHD = get_time();

    startInit = get_time();

    cudaMemset(d_conv, 0, imageSizeBytes);
    checkCUDAError("Initializing convolution result array");

    endInit = get_time();

    int blockDim = (int)sqrt(threadsPerBlock);

    blockDim = (blockDim > 32) ? 32 : blockDim;

    dim3 dimBlock(blockDim, blockDim);

    dim3 dimGrid((img_w + dimBlock.x - 1) / dimBlock.x,
                 (img_h + dimBlock.y - 1) / dimBlock.y);

    if (dimGrid.x > prop.maxGridSize[0]) {
        dimGrid.x = prop.maxGridSize[0]; // Limita al máximo permitido en x
    }

    if (dimGrid.y > prop.maxGridSize[1]) {
        dimGrid.y = prop.maxGridSize[1]; // Limita al máximo permitido en y
    }

    startConv = get_time();

    conv2dKernel<<<dimGrid, dimBlock>>>(d_image, d_conv, img_w, img_h, d_ker,
                                        ker_w, ker_h);

    cudaDeviceSynchronize();
    checkCUDAError("Executing convolution kernel");

    endConv = get_time();

    startDH = get_time();

    cudaMemcpy(h_conv, d_conv, imageSizeBytes, cudaMemcpyDeviceToHost);
    checkCUDAError("Copying convolution results to host");

    endDH = get_time();

    printf("Running sequential version for verification...\n");
    float* h_conv_seq = (float*)malloc(imageSizeBytes);
    if (h_conv_seq == NULL) {
        printf(
            "Error allocating host memory for sequential convolution result\n");
        return EXIT_FAILURE;
    }

    conv2d_sequential(h_image, img_w, img_h, h_conv_seq);

    int errors = 0;
    for (int i = 0; i < imageSize; i++) {
        if (fabs(h_conv[i] - h_conv_seq[i]) > 0.00001) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("Verification successful! Results match.\n");
    } else {
        printf("Verification failed! %d of %ld pixels differ.\n", errors,
               imageSize);
    }

#ifdef DEBUG
    savePGM32("output_sequential.pgm", h_conv_seq, img_w, img_h);
#endif

    int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, conv2dKernel,
                                                  blockDim * blockDim, 0);
    checkCUDAError("Calculating occupancy");

    float occupancy = (float)(maxBlocksPerSM * blockDim * blockDim) /
                      prop.maxThreadsPerMultiProcessor;

    overhead = startAlloc2 - startAlloc;
    double alloc_time = endAlloc - startAlloc2 - overhead;
    double init_time = endInit - startInit - overhead;
    double conv_time = endConv - startConv - overhead;
    double hd_time = endHD - startHD - overhead;
    double dh_time = endDH - startDH - overhead;
    double total_time = alloc_time + init_time + conv_time + hd_time + dh_time;

#ifdef DEBUG
    savePGM32("output.pgm", h_conv, img_w, img_h);
#endif

    printf("\nPAE,%s,%d,%d,%d,%f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%ld,"
           "PAE\n",
           imagePath, threadsPerBlock, dimGrid.x * dimGrid.y, maxBlocksPerSM,
           occupancy, overhead, alloc_time, init_time, conv_time, hd_time,
           dh_time, total_time, imageSize);

    cudaFree(d_image);
    checkCUDAError("Freeing device memory for image");

    cudaFree(d_conv);
    checkCUDAError("Freeing device memory for convolution result");

    cudaFree(d_ker);
    checkCUDAError("Freeing device memory for kernel");

    free(h_image);
    free(h_conv);
    free(h_conv_seq);

    return EXIT_SUCCESS;
}