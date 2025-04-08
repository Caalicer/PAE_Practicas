#include "headerImage.h"

baseType distance(baseType *data, baseType *centroid, baseType yi2, int bands) {

	baseType xiyi = 0;
	baseType xi2 = 0;

	for (int i = 0; i < bands; i++) {

		xiyi += data[i] * centroid[i];
		xi2 += data[i] * data[i];

	}

	return sqrt(xi2 - 2 * xiyi + yi2);

}

baseType* initCentroids(const HSI& data, int clusters) {

	baseType *centroids = (baseType *) malloc(clusters * data.bands * sizeof(baseType));

	for (int i = 0; i < clusters; i++) {

		int pos = rand() % data.slice;

		for (int j = 0; j < data.bands; j++) {

			centroids[i * data.bands + j] = data.x[pos * data.bands + j];

		}

	}

	return centroids;

}

void getYi2(baseType *yi2, baseType *centroids, int bands, int clusters) {

	for (int i = 0; i < clusters; i++) {

		yi2[i] = 0;

		for (int j = 0; j < bands; j++) {

			yi2[i] += centroids[i * bands + j] * centroids[i * bands + j];

		}

	}

}

u_char* computeKMeans(const HSI& data, int clusters, int iterations) {

	baseType *centroids = initCentroids(data, clusters);
	baseType *temp_centroids = (baseType *) malloc(clusters * data.bands * sizeof(baseType));
	baseType *yi2 = (baseType *) malloc(clusters * sizeof(baseType));

	int *count = (int *) malloc(clusters * sizeof(int));

	u_char* clustering = (u_char*) malloc(data.slice * sizeof(u_char));

	getYi2(yi2, centroids, data.bands, clusters);

	for (int _ = 0; _ < iterations; _++) {

		memset(count, 0, clusters * sizeof(int));

		for (int i = 0; i < data.slice; i++) {

			baseType min_dist = INFINITY;
			int min_index = -1;

			for (int j = 0; j < clusters; j++) {

				baseType dist = distance(&data.x[i * data.bands], &centroids[j * data.bands], yi2[j], data.bands);

				if (dist < min_dist) {

					min_dist = dist;
					min_index = j;

				}


			}

			clustering[i] = min_index;
			count[min_index]++;

		}

		if (_ == iterations - 1) {

			break;

		}

		memset(temp_centroids, 0, clusters * data.bands * sizeof(baseType));

		for (int i = 0; i < data.slice; i++) {

			for (int j = 0; j < data.bands; j++) {

				temp_centroids[clustering[i] * data.bands + j] += data.x[i * data.bands + j];

			}

		}

		for (int i = 0; i < clusters; i++) {

			if (count[i] == 0) {

				continue;

			}

			for (int j = 0; j < data.bands; j++) {

				centroids[i * data.bands + j] = temp_centroids[i * data.bands + j] / (float)count[i];

			}

		}

		getYi2(yi2, centroids, data.bands, clusters);

	}

	free(centroids);
	free(temp_centroids);
	free(yi2);
	free(count);

	return clustering;

}

int main(int argc, char *argv[]) {

	if (argc < 3) {

		printf("Use %s hyperspectral.raw format [num. clusters] [num. iterations]\n", argv[0]);
		printf(" * hyperspectral.raw - The hyperspectral data in pixel vector format\n");
		printf(" * number of clusters - (default 10)\n");
		printf(" * number of iterations - (default 500)\n");

		return EXIT_FAILURE;

	}

	int clusters = atoi( argv[2]);
	int iterations = atoi(argv[3]);

	HSI data;
	read_dataset_raw(data, argv[1]);

	data.standardize_data();

	u_char* clustering = computeKMeans(data, clusters, iterations);

	savePGM("clustering.pgm", clustering, data.cols, data.rows);
	data.free_memory();

	free(clustering);

	return EXIT_SUCCESS;

}

