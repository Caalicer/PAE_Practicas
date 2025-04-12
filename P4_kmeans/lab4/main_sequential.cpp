#include "headerImage.h"

#define DEFAULT_CLUSTERS 10
#define DEFAULT_MAX_ITERATIONS 500

typedef struct {
	double total;
	double overhead;
	double read;
	double standardize;
	double save;
	double free;
	double kmeans_total;
	double kmeans_malloc;
	double kmeans_init;
	double kmeans_yi2;
	double kmeans_distance;
	double kmeans_update;
	double kmeans_free;
	double kmeans_error;
	double kmeans_error_time;
	int clusters;
	int iterations;
	long image_size;
} ExecutionData;

double calculate_error(const HSI& data, u_char* clustering, baseType* centroids, baseType* yi2) {

	double total_error = 0.0;

	for (int i = 0; i < data.slice; i++) {

		int cluster_idx = clustering[i];

		baseType xiyi = 0;
		baseType xi2 = 0;

		for (int j = 0; j < data.bands; j++) {

			xiyi += data.x[i * data.bands + j] * centroids[cluster_idx * data.bands + j];
			xi2 += data.x[i * data.bands + j] * data.x[i * data.bands + j];

		}

		total_error += (xi2 - 2 * xiyi + yi2[cluster_idx]);

	}

	return total_error;

}

double get_time() {

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1.0e9;

}

baseType distance(baseType *data, baseType *centroid, baseType yi2, int bands) {

	baseType xiyi = 0;
	baseType xi2 = 0;

	for (int i = 0; i < bands; i++) {

		xiyi += data[i] * centroid[i];
		xi2 += data[i] * data[i];

	}

	return sqrt(xi2 - 2 * xiyi + yi2);

}

baseType* init_centroids(const HSI& data, int clusters) {

	baseType *centroids = (baseType *) malloc(clusters * data.bands * sizeof(baseType));

	if (!centroids) {

		printf("Error: Unable to alloc memory | Init centroids \n");
		return NULL;

	}

	for (int i = 0; i < clusters; i++) {

		int pos = rand() % data.slice;

		for (int j = 0; j < data.bands; j++) {

			centroids[i * data.bands + j] = data.x[pos * data.bands + j];

		}

	}

	return centroids;

}

void get_yi2(baseType *yi2, baseType *centroids, int bands, int clusters) {

	for (int i = 0; i < clusters; i++) {

		yi2[i] = 0;

		for (int j = 0; j < bands; j++) {

			yi2[i] += centroids[i * bands + j] * centroids[i * bands + j];

		}

	}

}

u_char* compute_kmeans(const HSI& data, int clusters, int iterations, ExecutionData* execution_data) {

	if (clusters <= 0 || clusters > data.slice) {

		printf("Error: Invalid clusters number \n");
		return NULL;

	}

	if (iterations <= 0) {

		printf("Error: Invalid iterations number \n");
		return NULL;

	}

	double kmeans_start = get_time();

	double malloc_start = get_time();

	baseType *temp_centroids = NULL;
	baseType *yi2 = NULL;
	int *count = NULL;
	u_char* clustering = NULL;

	temp_centroids = (baseType *) malloc(clusters * data.bands * sizeof(baseType));

	if (!temp_centroids) {

		printf("Error: Unable to alloc memory | temp centroids \n");

		exit(EXIT_FAILURE);

	}

	yi2 = (baseType *) malloc(clusters * sizeof(baseType));

	if (!yi2) {

		free(temp_centroids);

		printf("Error: Unable to alloc memory | yi2 \n");

		exit(EXIT_FAILURE);

	}

	count = (int *) malloc(clusters * sizeof(int));

	if (!count) {

		free(temp_centroids);
		free(yi2);

		printf("Error: Unable to alloc memory | count \n");

		exit(EXIT_FAILURE);

	}

	clustering = (u_char *) malloc(data.slice * sizeof(u_char));

	if (!count) {

		free(temp_centroids);
		free(yi2);
		free(count);

		printf("Error: Unable to alloc memory | clustering \n");

		exit(EXIT_FAILURE);

	}

	execution_data->kmeans_malloc = get_time() - malloc_start;

	#ifdef VERBOSE
		printf("Memory allocated in %.6f seconds\n", execution_data->kmeans_malloc);
	#endif

	double init_start = get_time();

	baseType *centroids = init_centroids(data, clusters);

	if (!centroids) {

		free(temp_centroids);
		free(yi2);
		free(count);
		free(clustering);

		printf("Error: Unable to init centroids | init_centroids \n");

		exit(EXIT_FAILURE);

	}

	get_yi2(yi2, centroids, data.bands, clusters);

	execution_data->kmeans_init = get_time() - init_start;

	#ifdef VERBOSE
		printf("Centroids initialized in %.6f seconds\n", execution_data->kmeans_init);
	#endif

	for (int iter = 0; iter < iterations; iter++) {

		#ifdef VERBOSE
			double iter_time = get_time();
			printf("Starting iteration %d/%d\n", iter+1, iterations);
		#endif

		double dist_start = get_time();

		memset(count, 0, clusters * sizeof(int));

		for (int i = 0; i < data.slice; i++) {

			baseType min_dist = INFINITY;
			int min_index = -1;

			for (int j = 0; j < clusters; j++) {

				baseType dist = distance(&data.x[i * data.bands],  &centroids[j * data.bands],  yi2[j], data.bands);

				if (dist < min_dist) {

					min_dist = dist;
					min_index = j;

				}

			}

			clustering[i] = min_index;
			count[min_index]++;

		}

		execution_data->kmeans_distance += get_time() - dist_start;

		if (iter == iterations - 1) {

			double yi2_start = get_time();

			get_yi2(yi2, centroids, data.bands, clusters);

			execution_data->kmeans_yi2 += get_time() - yi2_start;

			double error_start = get_time();

			execution_data->kmeans_error = calculate_error(data, clustering, centroids, yi2);

			execution_data->kmeans_error_time += get_time() - error_start;

			#ifdef VERBOSE
				printf("Iteration %d completed in %.6f seconds | Current error: %.6f \n", iter+1, get_time() - iter_time, execution_data->kmeans_error);
			#endif

			break;

		}

		double update_start = get_time();

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

		double yi2_start = get_time();

		get_yi2(yi2, centroids, data.bands, clusters);

		execution_data->kmeans_yi2 += get_time() - yi2_start;
		execution_data->kmeans_update += get_time() - update_start;

		double error_start = get_time();

		execution_data->kmeans_error = calculate_error(data, clustering, centroids, yi2);

		execution_data->kmeans_error_time += get_time() - error_start;

		#ifdef VERBOSE
			printf("Iteration %d completed in %.6f seconds | Current error: %.6f \n", iter+1, get_time() - iter_time, execution_data->kmeans_error);
		#endif

	}

	double free_start = get_time();

	free(centroids);
	free(temp_centroids);
	free(yi2);
	free(count);

	execution_data->kmeans_free = get_time() - free_start;

	#ifdef VERBOSE
		printf("Memory freed in %.6f seconds\n", execution_data->kmeans_free);
	#endif

	execution_data->kmeans_total = get_time() - kmeans_start;

	#ifdef VERBOSE
		printf("K-means completed in %.6f seconds\n", execution_data->kmeans_total);
	#endif

	return clustering;

}

int main(int argc, char *argv[]) {

	srand(time(NULL));

	if (argc < 2) {

		printf("Usage: %s hyperspectral.raw [num. clusters] [num. iterations]\n", argv[0]);
		printf(" * hyperspectral.raw - The hyperspectral data in pixel vector format\n");
		printf(" * number of clusters - (default %d)\n", DEFAULT_CLUSTERS);
		printf(" * number of iterations - (default %d)\n", DEFAULT_MAX_ITERATIONS);

		return EXIT_FAILURE;

	}

	const char* image_path = argv[1];

	ExecutionData execution_data = {0};

	execution_data.clusters = (argc < 3) ? DEFAULT_CLUSTERS : atoi(argv[2]);
	execution_data.iterations = (argc < 4) ? DEFAULT_MAX_ITERATIONS : atoi(argv[3]);

	if (execution_data.clusters <= 0) {

		printf("Error: Number of clusters must be positive \n");
		return EXIT_FAILURE;

	}

	if (execution_data.iterations <= 0) {

		printf("Error: Number of clusters must be positive \n");
		return EXIT_FAILURE;

	}

	double total_start = get_time();

	double overhead_start = get_time();
	double overhead_end = get_time();

	execution_data.overhead = overhead_end - overhead_start;

	double read_start = get_time();

	HSI data;

	read_dataset_raw(data, image_path);

	execution_data.read = get_time() - read_start - execution_data.overhead;

	execution_data.image_size = data.slice * data.bands * sizeof(baseType);

	#ifdef VERBOSE
		printf("Data read in %.6f seconds\n", execution_data.read);
	#endif

	double standarize_start = get_time();

	data.standardize_data();

	execution_data.standardize = get_time() - standarize_start - execution_data.overhead;

	#ifdef VERBOSE
		printf("Data standardized in %.6f seconds\n", execution_data.standardize);
	#endif

	u_char* clustering = compute_kmeans(data, execution_data.clusters, execution_data.iterations, &execution_data);

	if (!clustering) {

		data.free_memory();
		return EXIT_FAILURE;

	}

	double save_start = get_time();

	if (savePGM("sequential_result.pgm", clustering, data.cols, data.rows) != 0) {

		printf("Error: Unable to save result image \n");

		free(clustering);
		data.free_memory();

		return EXIT_FAILURE;

	}

	execution_data.save = get_time() - save_start - execution_data.overhead;

	#ifdef VERBOSE
		printf("Results saved in %.6f seconds\n", execution_data.save);
	#endif

	double free_start = get_time();

	data.free_memory();
	free(clustering);

	execution_data.free = get_time() - free_start - execution_data.overhead;

	execution_data.total = get_time() - total_start;

	#ifdef VERBOSE
		printf("Total execution time: %.6f seconds\n", execution_data.total);
	#endif

	printf("\nPAE,%s,%d,%ld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,PAE\n", image_path, execution_data.clusters, execution_data.image_size, execution_data.iterations, execution_data.overhead, execution_data.read, execution_data.standardize, execution_data.kmeans_malloc, execution_data.kmeans_init, execution_data.kmeans_yi2, execution_data.kmeans_distance, execution_data.kmeans_update, execution_data.kmeans_free, execution_data.kmeans_total, execution_data.kmeans_error, execution_data.kmeans_error_time, execution_data.save, execution_data.free, execution_data.total);

	return EXIT_SUCCESS;

}