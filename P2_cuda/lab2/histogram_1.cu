#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "pgmio.h"

#define GRAY_LEVELS 256

void checkCUDAError(const char *msg) {

	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {

		fprintf(stderr, "Cuda error: %s: %s in %s at line %d.\n", msg, cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);

	}

}

__global__ void histogram(unsigned char* input, int* histogram, size_t imageSize) {

	unsigned int localHistogram[GRAY_LEVELS] = {0};

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = tid; i < imageSize; i += stride) {
		localHistogram[input[i]]++;
	}

	for (int binIdx = 0; binIdx < GRAY_LEVELS; binIdx++) {

		if (localHistogram[binIdx] > 0) {

			atomicAdd(&histogram[binIdx], localHistogram[binIdx]);

		}

	}

}

int main(int argc, char** argv) {

	if (argc < 3) {

		fprintf(stderr, "Usage: %s <image.pgm> <threadsPerBlock>\n", argv[0]);
		return EXIT_FAILURE;

	}

	const char* imagePath = argv[1];
	int threadsPerBlock = atoi(argv[2]);

	struct timeval startAlloc, startAlloc2, endAlloc;
	struct timeval startInit, endInit;
	struct timeval startHistogram, endHistogram;
	struct timeval startHD, endHD;
	struct timeval startDH, endDH;

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

	gettimeofday(&startAlloc, NULL);
	gettimeofday(&startAlloc2, NULL);

	cudaMalloc(&d_image, imageSize * sizeof(unsigned char));
	checkCUDAError("Allocating device memory for image");

	cudaMalloc(&d_hist, GRAY_LEVELS * sizeof(int));
	checkCUDAError("Allocating device memory for histogram");

	gettimeofday(&endAlloc, NULL);

	gettimeofday(&startHD, NULL);

	cudaMemcpy(d_image, h_image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	checkCUDAError("Copying image data to device");

	gettimeofday(&endHD, NULL);

	gettimeofday(&startInit, NULL);

	cudaMemset(d_hist, 0, GRAY_LEVELS * sizeof(int));
	checkCUDAError("Initializing histogram array");

	gettimeofday(&endInit, NULL);

	int blocksPerGrid = (imageSize + threadsPerBlock - 1) / threadsPerBlock;

	if (blocksPerGrid > prop.maxGridSize[0]) {

		blocksPerGrid = prop.maxGridSize[0];

	}

	gettimeofday(&startHistogram, NULL);

	histogram<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_hist, imageSize);
	checkCUDAError("Executing optimized histogram kernel");

	gettimeofday(&endHistogram, NULL);

	gettimeofday(&startDH, NULL);

	cudaMemcpy(h_hist, d_hist, GRAY_LEVELS * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("Copying histogram results to host");

	gettimeofday(&endDH, NULL);

	int maxBlocksPerSM;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, histogram, threadsPerBlock, 0);
	checkCUDAError("Calculating occupancy");

	float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) / prop.maxThreadsPerMultiProcessor;

	double overhead = (startAlloc2.tv_sec - startAlloc.tv_sec) + (startAlloc2.tv_usec - startAlloc.tv_usec) / 1e6;
	double alloc_time = (endAlloc.tv_sec - startAlloc2.tv_sec) + (endAlloc.tv_usec - startAlloc2.tv_usec) / 1e6 - overhead;
	double init_time = (endInit.tv_sec - startInit.tv_sec) + (endInit.tv_usec - startInit.tv_usec) / 1e6 - overhead;
	double histogram_time = (endHistogram.tv_sec - startHistogram.tv_sec) + (endHistogram.tv_usec - startHistogram.tv_usec) / 1e6 - overhead;
	double hd_time = (endHD.tv_sec - startHD.tv_sec) + (endHD.tv_usec - startHD.tv_usec) / 1e6 - overhead;
	double dh_time = (endDH.tv_sec - startDH.tv_sec) + (endDH.tv_usec - startDH.tv_usec) / 1e6 - overhead;
	double total_time = alloc_time + init_time + histogram_time + hd_time + dh_time;

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

	printf("\nPAE,%s,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%ld,PAE\n", imagePath, threadsPerBlock, blocksPerGrid, maxBlocksPerSM, occupancy, overhead, alloc_time, init_time, histogram_time, hd_time, dh_time, total_time, imageSize);

	printf("First histogram value: %d\n", h_hist[0]);
	printf("Total histogram sum: %d\n", sum);

	cudaFree(d_image);
	checkCUDAError("Freeing device memory for image");

	cudaFree(d_hist);
	checkCUDAError("Freeing device memory for histogram");

	free(h_image);

	return EXIT_SUCCESS;

}