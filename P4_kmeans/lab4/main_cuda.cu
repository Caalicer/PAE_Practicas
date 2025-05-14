#include "headerImage.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DEFAULT_CLUSTERS 10
#define DEFAULT_MAX_ITERATIONS 500
#define BLOCK_SIZE 256

/**
 * @brief Macro para verificar errores de CUDA.
 *
 * Esta macro evalúa el resultado de una llamada a una función de la API de
 * CUDA. Si ocurre un error, imprime un mensaje con el archivo y la línea donde
 * ocurrió y termina la ejecución del programa.
 *
 * @param call Llamada a una función de la API de CUDA.
 */
#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

/**
 * @struct ExecutionData
 *
 * @brief Estructura para almacenar datos de ejecución relacionados con el
 * algoritmo k-means.
 *
 * Esta estructura contiene métricas de tiempo y estadísticas sobre el proceso
 * de ejecución del algoritmo k-means, incluyendo tiempos de inicialización,
 * transferencia de datos, ejecución y cálculo de errores.
 */
typedef struct {
    double total;            /**< Tiempo total de ejecución del programa. */
    double overhead;         /**< Tiempo de overhead al llamar a get_time(). */
    double read;             /**< Tiempo de lectura los datos de entrada. */
    double standardize;      /**< Tiempo de estandarización los datos. */
    double kmeans_total;     /**< Tiempo total dedicado al algoritmo k-means. */
    double kmeans_malloc;    /**< Tiempo dedicado a la reserva de memoria. */
    double kmeans_init;      /**< Tiempo de la inicialización de centroides. */
    double kmeans_transfer;  /**< Tiempo de transferencia de datos a la GPU. */
    double kmeans_execution; /**< Tiempo de ejecución principal del algoritmo.*/
    double kmeans_error; /**< Error total calculado al final del algoritmo. */
    int clusters;        /**< Número de clusters utilizados en k-means. */
    int iterations;      /**< Número de iteraciones realizadas. */
    long image_size;     /**< Tamaño de la imagen en bytes. */
} ExecutionData;

/**
 * @brief Obtener el tiempo actual en segundos con precisión de nanosegundos.
 *
 * Utiliza `clock_gettime` para obtener el tiempo actual en segundos.
 *
 * @return Tiempo actual en segundos como un valor de tipo `double`.
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
}

/**
 * @brief Kernel CUDA para inicializar los centroides aleatoriamente.
 *
 * Este kernel selecciona píxeles aleatorios de los datos de entrada como
 * centroides iniciales para el algoritmo k-means.
 *
 * @param data Datos de entrada (píxeles).
 * @param centroids Array donde se almacenarán los centroides iniciales.
 * @param bands Número de bandas espectrales.
 * @param clusters Número de clusters.
 * @param slice Número de píxeles en los datos.
 * @param seed Semilla para el generador de números aleatorios.
 */
__global__ void init_centroids_kernel(baseType* data, baseType* centroids,
                                      int bands, int clusters, int slice,
                                      unsigned int seed);

/**
 * @brief Kernel CUDA para calcular la norma al cuadrado de cada centroide.
 *
 * Este kernel calcula la norma al cuadrado de cada centroide y almacena
 * los resultados en un array.
 *
 * @param yi2 Array donde se almacenarán las normas al cuadrado.
 * @param centroids Array de centroides.
 * @param bands Número de bandas espectrales.
 * @param clusters Número de clusters.
 */
__global__ void get_yi2_kernel(baseType* yi2, baseType* centroids, int bands,
                               int clusters);

/**
 * @brief Kernel CUDA para asignar cada píxel al cluster más cercano.
 *
 * Este kernel calcula la distancia entre cada píxel y los centroides,
 * asignando cada píxel al cluster más cercano.
 *
 * @param data Datos de entrada (píxeles).
 * @param centroids Array de centroides.
 * @param yi2 Array con las normas al cuadrado de los centroides.
 * @param clustering Array donde se almacenarán las asignaciones de clusters.
 * @param bands Número de bandas espectrales.
 * @param clusters Número de clusters.
 * @param slice Número de píxeles en los datos.
 */
__global__ void assign_clusters_kernel(baseType* data, baseType* centroids,
                                       baseType* yi2, u_char* clustering,
                                       int bands, int clusters, int slice);

/**
 * @brief Kernel CUDA para calcular la suma de los puntos en cada cluster.
 *
 * Este kernel utiliza memoria compartida para acumular las sumas de los
 * puntos asignados a cada cluster, optimizando el acceso a memoria global.
 *
 * @param data Datos de entrada (píxeles).
 * @param clustering Array con las asignaciones de clusters.
 * @param temp_centroids Array temporal para almacenar las sumas.
 * @param count Array para contar el número de puntos en cada cluster.
 * @param bands Número de bandas espectrales.
 * @param clusters Número de clusters.
 * @param slice Número de píxeles en los datos.
 */
__global__ void sum_clusters_kernel(baseType* data, u_char* clustering,
                                    baseType* temp_centroids, int* count,
                                    int bands, int clusters, int slice);

/**
 * @brief Kernel CUDA para actualizar los centroides.
 *
 * Este kernel calcula los nuevos centroides dividiendo las sumas acumuladas
 * por el número de puntos en cada cluster.
 *
 * @param centroids Array de centroides.
 * @param temp_centroids Array temporal con las sumas acumuladas.
 * @param count Array con el número de puntos en cada cluster.
 * @param bands Número de bandas espectrales.
 * @param clusters Número de clusters.
 */
__global__ void update_centroids_kernel(baseType* centroids,
                                        baseType* temp_centroids, int* count,
                                        int bands, int clusters);

/**
 * @brief Kernel CUDA para calcular el error total.
 *
 * Este kernel calcula el error total como la suma de las distancias al cuadrado
 * entre cada píxel y su centroide asignado.
 *
 * @param data Datos de entrada (píxeles).
 * @param clustering Array con las asignaciones de clusters.
 * @param centroids Array de centroides.
 * @param yi2 Array con las normas al cuadrado de los centroides.
 * @param block_errors Array para almacenar los errores parciales por bloque.
 * @param bands Número de bandas espectrales.
 * @param slice Número de píxeles en los datos.
 */
__global__ void compute_error_kernel(baseType* data, u_char* clustering,
                                     baseType* centroids, baseType* yi2,
                                     double* block_errors, int bands,
                                     int slice);

/**
 * @brief Función para calcular el error total en la CPU.
 *
 * Suma los errores parciales calculados por los bloques en la GPU.
 *
 * @param block_errors Array con los errores parciales.
 * @param num_blocks Número de bloques utilizados.
 * @return Error total como un valor de tipo `double`.
 */
double calculate_error_from_blocks(double* block_errors, int num_blocks);

/**
 * @brief Implementación del algoritmo k-means utilizando CUDA.
 *
 * Esta función realiza el algoritmo k-means en la GPU, incluyendo la
 * inicialización de centroides, asignación de clusters, actualización
 * de centroides y cálculo del error total.
 *
 * @param data Datos de entrada en formato HSI.
 * @param clusters Número de clusters.
 * @param iterations Número de iteraciones.
 * @param execution_data Estructura para almacenar métricas de ejecución.
 * @return Array con las asignaciones de clusters para cada píxel.
 */
u_char* compute_kmeans_cuda(const HSI& data, int clusters, int iterations,
                            ExecutionData* execution_data);

/**
 * @brief Función principal del programa.
 *
 * Lee los datos de entrada, ejecuta el algoritmo k-means en la GPU y guarda
 * los resultados en un archivo de salida.
 *
 * @param argc Número de argumentos de línea de comandos.
 * @param argv Array de argumentos de línea de comandos.
 * @return Código de salida del programa.
 */
int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: %s hyperspectral.raw [num. clusters] [num. "
               "iterations]\n",
               argv[0]);
        printf(" * hyperspectral.raw - The hyperspectral data in pixel "
               "vector "
               "format\n");
        printf(" * number of clusters - (default %d)\n", DEFAULT_CLUSTERS);
        printf(" * number of iterations - (default %d)\n",
               DEFAULT_MAX_ITERATIONS);

        return EXIT_FAILURE;
    }

    const char* image_path = argv[1];

    ExecutionData execution_data = {0};

    execution_data.clusters = (argc < 3) ? DEFAULT_CLUSTERS : atoi(argv[2]);
    execution_data.iterations =
        (argc < 4) ? DEFAULT_MAX_ITERATIONS : atoi(argv[3]);

    if (execution_data.clusters <= 0) {
        printf("Error: Number of clusters must be positive\n");
        return EXIT_FAILURE;
    }

    if (execution_data.iterations <= 0) {
        printf("Error: Number of iterations must be positive\n");
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

    execution_data.standardize =
        get_time() - standarize_start - execution_data.overhead;

#ifdef VERBOSE
    printf("Data standardized in %.6f seconds\n", execution_data.standardize);
#endif

    u_char* clustering =
        compute_kmeans_cuda(data, execution_data.clusters,
                            execution_data.iterations, &execution_data);

    if (!clustering) {
        data.free_memory();
        return EXIT_FAILURE;
    }

    if (savePGM("cuda_result.pgm", clustering, data.cols, data.rows) != 0) {
        printf("Error: Unable to save result image\n");

        free(clustering);
        data.free_memory();

        return EXIT_FAILURE;
    }

    data.free_memory();
    free(clustering);

    execution_data.total = get_time() - total_start;

#ifdef VERBOSE
    printf("Total execution time: %.6f seconds\n", execution_data.total);
#endif

    printf("\nPAE,%s,%d,%ld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%."
           "6f,PAE\n",
           image_path, execution_data.clusters, execution_data.image_size,
           execution_data.iterations, execution_data.overhead,
           execution_data.read, execution_data.standardize,
           execution_data.kmeans_malloc, execution_data.kmeans_init,
           execution_data.kmeans_transfer, execution_data.kmeans_execution,
           execution_data.kmeans_error, execution_data.total);

    return EXIT_SUCCESS;
}

// Kernel CUDA para inicializar los centroides aleatoriamente.
__global__ void init_centroids_kernel(baseType* data, baseType* centroids,
                                      int bands, int clusters, int slice,
                                      unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < clusters) {
        curandState state;
        curand_init(seed, tid, 0, &state);

        // Seleccionar un pixel aleatorio como centroide
        int pos = curand(&state) % slice;

        for (int j = 0; j < bands; j++) {
            centroids[tid * bands + j] = data[pos * bands + j];
        }
    }
}

// Kernel CUDA para calcular la norma al cuadrado de cada centroide.
__global__ void get_yi2_kernel(baseType* yi2, baseType* centroids, int bands,
                               int clusters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < clusters) {
        yi2[tid] = 0;

        for (int j = 0; j < bands; j++) {
            yi2[tid] += centroids[tid * bands + j] * centroids[tid * bands + j];
        }
    }
}

// Kernel CUDA para asignar cada píxel al cluster más cercano.
__global__ void assign_clusters_kernel(baseType* data, baseType* centroids,
                                       baseType* yi2, u_char* clustering,
                                       int bands, int clusters, int slice) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < slice) {
        baseType min_dist = INFINITY;
        int min_index = -1;

        for (int j = 0; j < clusters; j++) {
            baseType xiyi = 0;
            baseType xi2 = 0;

            for (int b = 0; b < bands; b++) {
                xiyi += data[tid * bands + b] * centroids[j * bands + b];
                xi2 += data[tid * bands + b] * data[tid * bands + b];
            }

            baseType dist = sqrt(xi2 - 2 * xiyi + yi2[j]);

            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }

        clustering[tid] = min_index;
    }
}

// Kernel CUDA para calcular la suma de los puntos en cada cluster.
__global__ void sum_clusters_kernel(baseType* data, u_char* clustering,
                                    baseType* temp_centroids, int* count,
                                    int bands, int clusters, int slice) {
    extern __shared__ int shared_counts[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Inicializar memoria compartida
    for (int i = local_tid; i < clusters; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    if (tid < slice) {
        int cluster_idx = clustering[tid];
        atomicAdd(&shared_counts[cluster_idx], 1);

        for (int j = 0; j < bands; j++) {
            atomicAdd(&temp_centroids[cluster_idx * bands + j],
                      data[tid * bands + j]);
        }
    }
    __syncthreads();

    // Posible mejora con menos accesos a memoria global
    // Acumular contadores de la memoria compartida a la global
    for (int i = local_tid; i < clusters; i += blockDim.x) {
        if (shared_counts[i] > 0) {
            atomicAdd(&count[i], shared_counts[i]);
        }
    }
}

// Kernel CUDA para actualizar los centroides.
__global__ void update_centroids_kernel(baseType* centroids,
                                        baseType* temp_centroids, int* count,
                                        int bands, int clusters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < clusters * bands) {
        int cluster_idx = tid / bands;
        int band_idx = tid % bands;

        if (count[cluster_idx] > 0) {
            centroids[tid] = temp_centroids[tid] / count[cluster_idx];
        }
    }
}

// Kernel CUDA para calcular el error total.
__global__ void compute_error_kernel(baseType* data, u_char* clustering,
                                     baseType* centroids, baseType* yi2,
                                     double* block_errors, int bands,
                                     int slice) {
    extern __shared__ double shared_errors[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    shared_errors[local_tid] = 0.0;

    if (tid < slice) {
        int cluster_idx = clustering[tid];

        baseType xiyi = 0;
        baseType xi2 = 0;

        for (int j = 0; j < bands; j++) {
            xiyi += data[tid * bands + j] * centroids[cluster_idx * bands + j];
            xi2 += data[tid * bands + j] * data[tid * bands + j];
        }

        shared_errors[local_tid] = xi2 - 2 * xiyi + yi2[cluster_idx];
    }
    __syncthreads();

    // Reducción en memoria compartida
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_errors[local_tid] += shared_errors[local_tid + stride];
        }
        __syncthreads();
    }

    // El primer hilo de cada bloque escribe el resultado
    if (local_tid == 0) {
        block_errors[blockIdx.x] = shared_errors[0];
    }
}

// Función para calcular el error total en la CPU./
double calculate_error_from_blocks(double* block_errors, int num_blocks) {
    double total_error = 0.0;

    for (int i = 0; i < num_blocks; i++) {
        total_error += block_errors[i];
    }

    return total_error;
}

// Implementación del algoritmo k-means utilizando CUDA.
u_char* compute_kmeans_cuda(const HSI& data, int clusters, int iterations,
                            ExecutionData* execution_data) {
    if (clusters <= 0 || clusters > data.slice) {
        printf("Error: Invalid clusters number\n");
        return NULL;
    }

    if (iterations <= 0) {
        printf("Error: Invalid iterations number\n");
        return NULL;
    }

    double kmeans_start = get_time();

    // Variables en host
    baseType* d_data = NULL;
    baseType* d_centroids = NULL;
    baseType* d_temp_centroids = NULL;
    baseType* d_yi2 = NULL;
    int* d_count = NULL;
    u_char* d_clustering = NULL;
    double* d_block_errors = NULL;

    u_char* h_clustering = NULL;
    double* h_block_errors = NULL;

    double malloc_start = get_time();

    // Reserva de memoria en el host para el resultado final
    h_clustering = (u_char*)malloc(data.slice * sizeof(u_char));
    if (!h_clustering) {
        printf("Error: Unable to allocate memory for clustering result "
               "on host\n");
        return NULL;
    }

    // Calcular dimensiones de grid y bloques
    int num_blocks_pixels = (data.slice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_clusters = (clusters + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_centroids =
        (clusters * data.bands + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Reserva de memoria para el cálculo de error
    h_block_errors = (double*)malloc(num_blocks_pixels * sizeof(double));
    if (!h_block_errors) {
        printf("Error: Unable to allocate memory for block errors on "
               "host\n");
        free(h_clustering);
        return NULL;
    }

    // Reserva de memoria en device
    CUDA_CHECK(cudaMalloc((void**)&d_data,
                          data.slice * data.bands * sizeof(baseType)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids,
                          clusters * data.bands * sizeof(baseType)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp_centroids,
                          clusters * data.bands * sizeof(baseType)));
    CUDA_CHECK(cudaMalloc((void**)&d_yi2, clusters * sizeof(baseType)));
    CUDA_CHECK(cudaMalloc((void**)&d_count, clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_clustering, data.slice * sizeof(u_char)));
    CUDA_CHECK(cudaMalloc((void**)&d_block_errors,
                          num_blocks_pixels * sizeof(double)));

    execution_data->kmeans_malloc = get_time() - malloc_start;

#ifdef VERBOSE
    printf("Memory allocated in %.6f seconds\n", execution_data->kmeans_malloc);
#endif

    double transfer_start = get_time();

    // Transferir datos a la GPU
    CUDA_CHECK(cudaMemcpy(d_data, data.x,
                          data.slice * data.bands * sizeof(baseType),
                          cudaMemcpyHostToDevice));

    execution_data->kmeans_transfer = get_time() - transfer_start;

#ifdef VERBOSE
    printf("Data transferred to GPU in %.6f seconds\n",
           execution_data->kmeans_transfer);
#endif

    double init_start = get_time();

    // Inicializar centroides
    init_centroids_kernel<<<num_blocks_clusters, BLOCK_SIZE>>>(
        d_data, d_centroids, data.bands, clusters, data.slice, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calcular yi2 (norma al cuadrado de cada centroide)
    get_yi2_kernel<<<num_blocks_clusters, BLOCK_SIZE>>>(d_yi2, d_centroids,
                                                        data.bands, clusters);
    CUDA_CHECK(cudaDeviceSynchronize());

    execution_data->kmeans_init = get_time() - init_start;

#ifdef VERBOSE
    printf("Centroids initialized in %.6f seconds\n",
           execution_data->kmeans_init);
#endif

    double execution_start = get_time();

    // Bucle principal de k-means
    for (int iter = 0; iter < iterations; iter++) {
#ifdef VERBOSE
        double iter_time = get_time();
        printf("Starting iteration %d/%d\n", iter + 1, iterations);
#endif

        // Asignar cada punto al cluster más cercano
        assign_clusters_kernel<<<num_blocks_pixels, BLOCK_SIZE>>>(
            d_data, d_centroids, d_yi2, d_clustering, data.bands, clusters,
            data.slice);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reiniciar contadores y sumas para actualizar centroides
        CUDA_CHECK(cudaMemset(d_count, 0, clusters * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_temp_centroids, 0,
                              clusters * data.bands * sizeof(baseType)));

        // Sumar puntos por cluster para actualizar centroides
        sum_clusters_kernel<<<num_blocks_pixels, BLOCK_SIZE,
                              clusters * sizeof(int)>>>(
            d_data, d_clustering, d_temp_centroids, d_count, data.bands,
            clusters, data.slice);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Actualizar centroides
        update_centroids_kernel<<<num_blocks_centroids, BLOCK_SIZE>>>(
            d_centroids, d_temp_centroids, d_count, data.bands, clusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Recalcular yi2
        get_yi2_kernel<<<num_blocks_clusters, BLOCK_SIZE>>>(
            d_yi2, d_centroids, data.bands, clusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        // En la última iteración, calcular error total
        if (iter == iterations - 1) {
            compute_error_kernel<<<num_blocks_pixels, BLOCK_SIZE,
                                   BLOCK_SIZE * sizeof(double)>>>(
                d_data, d_clustering, d_centroids, d_yi2, d_block_errors,
                data.bands, data.slice);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Transferir resultados parciales del error al host
            CUDA_CHECK(cudaMemcpy(h_block_errors, d_block_errors,
                                  num_blocks_pixels * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            // Calcular error total en el host
            execution_data->kmeans_error =
                calculate_error_from_blocks(h_block_errors, num_blocks_pixels);

#ifdef VERBOSE
            printf("Iteration %d completed | Current error: %.6f\n", iter + 1,
                   execution_data->kmeans_error);
#endif

            // Transferir resultado de clustering al host
            CUDA_CHECK(cudaMemcpy(h_clustering, d_clustering,
                                  data.slice * sizeof(u_char),
                                  cudaMemcpyDeviceToHost));
        }
#ifdef VERBOSE
        else {
            printf("Iteration %d completed\n", iter + 1);
        }
#endif
    }

    execution_data->kmeans_execution = get_time() - execution_start;

#ifdef VERBOSE
    printf("K-means execution completed in %.6f seconds\n",
           execution_data->kmeans_execution);
#endif

    // Liberar memoria en el dispositivo
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_temp_centroids);
    cudaFree(d_yi2);
    cudaFree(d_count);
    cudaFree(d_clustering);
    cudaFree(d_block_errors);

    free(h_block_errors);

    execution_data->kmeans_total = get_time() - kmeans_start;

#ifdef VERBOSE
    printf("K-means total time: %.6f seconds\n", execution_data->kmeans_total);
#endif

    return h_clustering;
}