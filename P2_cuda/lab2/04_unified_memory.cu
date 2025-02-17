// includes, system
#include <stdio.h>
#include <assert.h>

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	// pointer and dimension for host memory
	int n, dim = 262144;
	float *h_a, *h_b;

	// pointers for device memory
	float *d_a;

	size_t size = dim * sizeof(float);

	// allocate unified memory for h_a and d_a;
	cudaMallocManaged(&h_a, size);
	cudaMallocManaged(&h_b, size);
	cudaMallocManaged(&d_a, size);

	// initialize input data in host
	for (n = 0; n < dim; n++) {
		h_a[n] = (float)n;
	}

	// copy data from h_a to d_a using a loop in CPU
	for (n = 0; n < dim; n++) {
		d_a[n] = h_a[n];
	}

	// device to host copy h_b = d_a
	cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost);

	// verify the data on the host is correct
	for (n = 0; n < dim; n++) {
		assert(h_b[n] == h_a[n]);
	}

	// free all memory
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(d_a);

	// If the program makes it this far, then the results are correct and
	// there are no run-time errors.  Good work!
	printf("Correct!\n");

	return 0;

}

// nvcc 04_unified_memory.cu - 04_unified_memory
