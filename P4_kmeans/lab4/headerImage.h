#ifndef HEADERIMAGE_H_
#define HEADERIMAGE_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <limits.h>
#include <unistd.h>
#include <assert.h>

typedef float baseType;
typedef unsigned char uchar;

struct HSI {

	// data
	baseType* x;

	// dimensions
	int cols, rows, bands;

	// num of pixel in a band = cols*rows
	int slice;

	// num of elements = cols*rows*bands
	int size;

	HSI():x(NULL),cols(0),rows(0),bands(0),slice(0),size(0){ }

	void free_memory() {

		if (x != NULL) {

			free(x);

		}

	}

	void print(const char* varname) {

		printf(" %s:\n", varname);
		printf(" x   [%p]\n", x);
		printf(" dim: %d x %d x %d\n", cols, rows, bands);

	}

	void standardize_data() {

		baseType mean = 0;
		baseType stdev = 0;

		for (int i = 0; i < size; i++) {

			mean += x[i];

		}

		mean /= size;

		for (int i = 0; i < size; i++) {

			stdev += (x[i] - mean) * (x[i] - mean);

		}

		stdev = sqrt(stdev / size);

		for (int i = 0; i < size; i++) {

			x[i] = (x[i] - mean) / stdev;

		}

	}

};

void read_dataset_raw(HSI &data, const char* filename) {

	FILE *fp;
	size_t a;

	fp = fopen(filename, "r");

	if(fp == NULL) {

		printf("Can not open file %s\n", filename);
		exit(-1);

	}

	printf(" Open file %s\n", filename);

	a=fread(&data.bands, 4, 1, fp);
	a=fread(&data.cols, 4, 1, fp);
	a=fread(&data.rows, 4, 1, fp);

	data.size = data.cols * data.rows * data.bands;
	data.slice = data.cols * data.rows;

	int *aux = (int*) calloc (data.size, sizeof(int));
	data.x = (baseType*) calloc (data.size, sizeof(baseType));

	a=fread(aux, 4, data.size, fp);

	if(a != (size_t) data.size) {

		fprintf(stderr, "NOT ENOUGH DATA\n"); exit(-1);

	}

	printf(" Read %lu bytes\n", data.size * sizeof(baseType));

	int i;

	for ( i = 0; i < data.size; i++) {

		data.x[i]=(baseType)aux[i];

	}

	free(aux);
	fclose(fp);

}

int savePGM(const char *filename, uchar* v, int width, int height) {

	FILE *file;
	uchar c;
	int i;

	unsigned char A = v[0], B = v[0];

	for (i = 0; i < width * height; ++i) {

		A = (v[i] < A) ? v[i] : A;
		B = (v[i] > B) ? v[i] : B;

	}

	assert(A != B);

	for (i = 0; i < width * height; ++i) {

		v[i] = (unsigned) floor(((v[i] - A) * (255) / (B - A)));

	}

	file = fopen((char*) filename, "w");

	if (file == NULL) {

		perror("Saving PGM: Can not write to file ");
		exit(1);

	}

	fprintf(file, "P5\n# CREATOR: pQB\n%d %d\n255\n", width, height);

	for(i = 0; i < width * height; ++i) {

		c = v[i];
		fputc(c, file);

	}

	fclose(file);

	printf("[info] Saved '%s', %d x %d (%i pixels) (%li Kb)\n", filename, width, height, width * height, (width * height * sizeof(unsigned char)) / 1024);

	return 0;

}

#endif /* HEADERIMAGE_H_ */
