#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "pgmio.h"

#define GRAY_LEVELS 256
#define DEFAULT_PIXELS_PER_THREAD 4

void checkCUDAError(const char *msg) {

	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {

		fprintf(stderr, "Cuda error: %s: %s in %s at line %d.\n", msg, cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);

	}

}

__global__ void histogramGlobalCoalesced(unsigned char* image, int* hist, size_t imageSize, int pixelsPerThread) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;

	for (int i = 0; i < pixelsPerThread; i++) {

		int pixelIdx = tid + i * numThreads;

		if (pixelIdx < imageSize) {

			unsigned char pixel = image[pixelIdx];
			atomicAdd(&hist[pixel], 1);

		}

	}

}

int main(int argc, char** argv) {

	if (argc < 4) {

		fprintf(stderr, "Usage: %s <image.pgm> <blockSize> [pixelsPerThread]\n", argv[0]);
		return EXIT_FAILURE;

	}

	const char* imagePath = argv[1];
	int blockSize = atoi(argv[2]);
	int pixelsPerThread = (argc > 3) ? atoi(argv[3]) : DEFAULT_PIXELS_PER_THREAD;

	struct timeval startAlloc, startAlloc2, endAlloc;
	struct timeval startInit, endInit;
	struct timeval startHistogram, endHistogram;

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
	cudaDeviceSynchronize();
	checkCUDAError("Getting device properties");

	gettimeofday(&startAlloc, NULL);
	gettimeofday(&startAlloc2, NULL);

	cudaMalloc(&d_image, imageSize * sizeof(unsigned char));
	cudaDeviceSynchronize();
	checkCUDAError("Allocating device memory for image");

	cudaMalloc(&d_hist, GRAY_LEVELS * sizeof(int));
	cudaDeviceSynchronize();
	checkCUDAError("Allocating device memory for histogram");

	gettimeofday(&endAlloc, NULL);

	gettimeofday(&startInit, NULL);

	cudaMemcpy(d_image, h_image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCUDAError("Copying image data to device");

	cudaMemset(d_hist, 0, GRAY_LEVELS * sizeof(int));
	cudaDeviceSynchronize();
	checkCUDAError("Initializing histogram array");

	gettimeofday(&endInit, NULL);

	int threadsPerBlock = blockSize;
	int blocksPerGrid = (imageSize + threadsPerBlock * pixelsPerThread - 1) / (threadsPerBlock * pixelsPerThread);

	if (blocksPerGrid > prop.maxGridSize[0]) {

		blocksPerGrid = prop.maxGridSize[0];

	}

	gettimeofday(&startHistogram, NULL);

	histogramGlobalCoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_hist, imageSize, pixelsPerThread);
	cudaDeviceSynchronize();
	checkCUDAError("Executing histogram kernel");

	gettimeofday(&endHistogram, NULL);

	cudaMemcpy(h_hist, d_hist, GRAY_LEVELS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	checkCUDAError("Copying histogram results to host");

	int maxBlocksPerSM;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, histogramGlobalCoalesced, threadsPerBlock, 0);
	cudaDeviceSynchronize();
	checkCUDAError("Calculating occupancy");

	float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) / prop.maxThreadsPerMultiProcessor;

	double overhead = (startAlloc2.tv_sec - startAlloc.tv_sec) + (startAlloc2.tv_usec - startAlloc.tv_usec) / 1e6;
	double alloc_time = (endAlloc.tv_sec - startAlloc2.tv_sec) + (endAlloc.tv_usec - startAlloc2.tv_usec) / 1e6 - overhead;
	double init_time = (endInit.tv_sec - startInit.tv_sec) + (endInit.tv_usec - startInit.tv_usec) / 1e6;
	double histogram_time = (endHistogram.tv_sec - startHistogram.tv_sec) + (endHistogram.tv_usec - startHistogram.tv_usec) / 1e6;

	#ifdef DEBUG

		for (int i = 0; i < GRAY_LEVELS; i++) {

			printf("%d, %d\n", i, h_hist[i]);

		}

	#endif

	int sum = 0;

	for (int i = 0; i < GRAY_LEVELS; i++) {

		sum += h_hist[i];

	}

	if (sum != imageSize) {

		printf("Warning: Histogram sum (%d) does not match image size (%ld)\n", sum, imageSize);

	}

	printf("\nPAE,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%ld,%d\n", imagePath, blockSize, -1, blocksPerGrid, threadsPerBlock, occupancy, overhead, alloc_time, init_time, histogram_time, imageSize, pixelsPerThread);

	printf("First histogram value: %d\n", h_hist[0]);
	printf("Total histogram sum: %d\n", sum);

	cudaFree(d_image);
	cudaDeviceSynchronize();
	checkCUDAError("Freeing device memory for image");

	cudaFree(d_hist);
	cudaDeviceSynchronize();
	checkCUDAError("Freeing device memory for histogram");

	free(h_image);

	return EXIT_SUCCESS;

}