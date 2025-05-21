#include "headerImage.h"
#include <cstdlib>
#include <cuda_runtime.h>

#define DEFAULT_CLUSTERS 10
#define DEFAULT_MAX_ITERATIONS 500
#define DEFAULT_BLOCK_SIZE 256

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
 *
 * @var total Tiempo total de ejecución del programa.
 * @var overhead Tiempo de llamada a funciones de medición.
 * @var read Tiempo de lectura de la imagen.
 * @var standardize Tiempo de normalización estadística.
 * @var save Tiempo de guardado de la imagen PGM.
 * @var free Tiempo de liberación de la memoria.
 * @var kmeans_total Tiempo consumido por el núcleo del algoritmo.
 * @var kmeans_malloc Tiempo dedicado a la reserva de memoria.
 * @var kmeans_init Tiempo de la inicialización de centroides.
 * @var kmeans_yi2_init Tiempo del cálculo inicial de yi2.
 * @var kmeans_yi2 Tiempo de cálculo de yi2 acumulado.
 * @var kmeans_distance Tiempo de cálculo de distancias.
 * @var kmeans_update_1 Tiempo de acumulación de valores espectrales.
 * @var kmeans_update_2 Tiempo de normalización de valores acumulados.
 * @var kmeans_free Tiempo de liberación de estructuras.
 * @var kmeans_transfer Tiempo de transferencia de datos a la GPU.
 * @var kmeans_execution Tiempo de ejecución principal del algoritmo.
 * @var kmeans_error Error total calculado al final del algoritmo.
 * @var clusters Número de clusters utilizados en k-means.
 * @var iterations Número de iteraciones realizadas.
 * @var block_size Tamaño del bloque utilizado en CUDA.
 * @var image_size Tamaño de la imagen en bytes.
 * @var maxBlocksPerSM_distance Máximo número de bloques activos por
 *                              multiprocesador para el kernel de cálculo de
 *                              distancias.
 * @var maxBlocksPerSM_sum Máximo número de bloques activos por multiprocesador
 *                         para el kernel de suma de clusters.
 * @var maxBlocksPerSM_update Máximo número de bloques activos por
 *                            multiprocesador para el kernel de actualización de
 *                            centroides.
 * @var maxBlocksPerSM_yi2 Máximo número de bloques activos por multiprocesador
 *                         para el kernel de cálculo de yi2.
 * @var maxBlocksPerSM_error Máximo número de bloques activos por
 *                           multiprocesador para el kernel de cálculo de error.
 * @var occupancy_distance Ocupación del kernel de cálculo de distancias.
 * @var occupancy_sum Ocupación del kernel de suma de clusters.
 * @var occupancy_update Ocupación del kernel de actualización de centroides.
 * @var occupancy_yi2 Ocupación del kernel de cálculo de yi2.
 * @var occupancy_error Ocupación del kernel de cálculo de error.
 */
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
    double kmeans_yi2_init;
    double kmeans_yi2;
    double kmeans_distance;
    double kmeans_update_1;
    double kmeans_update_2;
    double kmeans_free;
    double kmeans_transfer;
    double kmeans_execution;
    double kmeans_error;
    int clusters;
    int iterations;
    int block_size;
    long image_size;
    int maxBlocksPerSM_distance;
    int maxBlocksPerSM_sum;
    int maxBlocksPerSM_update;
    int maxBlocksPerSM_yi2;
    int maxBlocksPerSM_error;
    float occupancy_distance;
    float occupancy_sum;
    float occupancy_update;
    float occupancy_yi2;
    float occupancy_error;
} ExecutionData;

/**
 * @brief Obtener el tiempo actual en segundos con precisión de
 * nanosegundos.
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
 * @brief Inicializa los centroides aleatoriamente..
 *
 * @param data Datos de entrada (píxeles).
 * @param clusters Número de clusters.
 * @return baseType* Array de centroides iniciales.
 */
baseType* init_centroids(const HSI& data, int clusters);

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
 * @param clusters Número de clusters
 * @param slice Número de píxeles en los datos.
 */
__global__ void compute_error_kernel(baseType* data, u_char* clustering,
                                     baseType* centroids, baseType* yi2,
                                     float* total_error, int bands,
                                     int clusters, int slice);

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

    srand(0);

    if (argc < 2) {
        printf("Usage: %s hyperspectral.raw [block size] [num. clusters] [num. "
               "iterations]\n",
               argv[0]);
        printf(" * hyperspectral.raw - The hyperspectral data in pixel "
               "vector "
               "format\n");
        printf(" * block size - (default %d)\n", DEFAULT_BLOCK_SIZE);
        printf(" * number of clusters - (default %d)\n", DEFAULT_CLUSTERS);
        printf(" * number of iterations - (default %d)\n",
               DEFAULT_MAX_ITERATIONS);

        return EXIT_FAILURE;
    }

    char* image_path = argv[1];

    ExecutionData execution_data = {0};

    execution_data.block_size = (argc < 3) ? DEFAULT_BLOCK_SIZE : atoi(argv[2]);

    execution_data.clusters = (argc < 4) ? DEFAULT_CLUSTERS : atoi(argv[3]);

    execution_data.iterations =
        (argc < 5) ? DEFAULT_MAX_ITERATIONS : atoi(argv[4]);

    if (execution_data.block_size <= 0) {
        printf("Error: Block size must be positive\n");
        return EXIT_FAILURE;
    }

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

#ifdef SAVE
    char* image_name = strrchr(image_path, '/');
    image_name++;

    char output_path[256];

    snprintf(output_path, sizeof(output_path), "./outs/result_%s_%d_%d_%d.pgm",
             image_name, execution_data.clusters, execution_data.iterations,
             execution_data.block_size);

    double save_start = get_time();

    if (savePGM(output_path, clustering, data.cols, data.rows) != 0) {

        printf("Error: Unable to save result image\n");

        free(clustering);
        data.free_memory();

        return EXIT_FAILURE;
    }

    execution_data.save = get_time() - save_start - execution_data.overhead;
#endif

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

    printf(
        "\nPAE,%s,%d,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%."
        "6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,PAE\n",
        image_path, execution_data.block_size, execution_data.clusters,
        execution_data.image_size, execution_data.overhead, execution_data.read,
        execution_data.standardize, execution_data.kmeans_malloc,
        execution_data.kmeans_init, execution_data.kmeans_yi2_init,
        execution_data.kmeans_yi2, execution_data.kmeans_distance,
        execution_data.kmeans_update_1, execution_data.kmeans_update_2,
        execution_data.kmeans_free, execution_data.kmeans_transfer,
        execution_data.kmeans_execution, execution_data.kmeans_total,
        execution_data.kmeans_error, execution_data.save, execution_data.free,
        execution_data.total, execution_data.iterations,
        execution_data.maxBlocksPerSM_distance,
        execution_data.maxBlocksPerSM_sum, execution_data.maxBlocksPerSM_update,
        execution_data.maxBlocksPerSM_yi2, execution_data.maxBlocksPerSM_error,
        execution_data.occupancy_distance, execution_data.occupancy_sum,
        execution_data.occupancy_update, execution_data.occupancy_yi2,
        execution_data.occupancy_error);

    return EXIT_SUCCESS;
}

// Inicializar los centroides aleatoriamente.
baseType* init_centroids(const HSI& data, int clusters) {

    baseType* centroids =
        (baseType*)malloc(clusters * data.bands * sizeof(baseType));

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

    extern __shared__ baseType shared_mem[]; // Memoria compartida dinámica

    // Particionar la memoria compartida
    baseType* shared_centroids = shared_mem; // [clusters * bands]
    baseType* shared_yi2 =
        (baseType*)&shared_mem[clusters * bands]; // [clusters]

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Cargar centroides en memoria compartida de forma colaborativa
    for (int i = local_tid; i < clusters * bands; i += blockDim.x) {
        shared_centroids[i] = centroids[i];
    }

    // Cargar yi2 en memoria compartida
    if (local_tid < clusters) {
        shared_yi2[local_tid] = yi2[local_tid];
    }

    __syncthreads();

    if (tid < slice) {
        baseType min_dist = INFINITY;
        int min_index = -1;

        for (int j = 0; j < clusters; j++) {
            baseType xiyi = 0;
            baseType xi2 = 0;

            for (int b = 0; b < bands; b++) {
                xiyi += data[tid * bands + b] * shared_centroids[j * bands + b];
                xi2 += data[tid * bands + b] * data[tid * bands + b];
            }

            baseType dist = sqrt(xi2 - 2 * xiyi + shared_yi2[j]);

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
    // Definir memoria compartida para acumular valores por bloque
    extern __shared__ char shared[];
    baseType* shared_sums = (baseType*)shared;
    int* shared_counts = (int*)(shared + sizeof(baseType) * clusters * bands);

    // Inicializar memoria compartida
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Inicializar memoria compartida
    for (int i = local_tid; i < clusters * bands; i += blockDim.x) {
        shared_sums[i] = 0;
    }

    for (int i = local_tid; i < clusters; i += blockDim.x) {
        shared_counts[i] = 0;
    }

    __syncthreads();

    // Acumular en memoria compartida
    if (tid < slice) {
        int cluster_idx = clustering[tid];
        atomicAdd(&shared_counts[cluster_idx], 1);

        for (int j = 0; j < bands; j++) {
            atomicAdd(&shared_sums[cluster_idx * bands + j],
                      data[tid * bands + j]);
        }
    }

    __syncthreads();

    // Acumular de memoria compartida a global
    for (int i = local_tid; i < clusters; i += blockDim.x) {
        if (shared_counts[i] > 0) {
            atomicAdd(&count[i], shared_counts[i]);
        }

        for (int j = 0; j < bands; j++) {
            atomicAdd(&temp_centroids[i * bands + j],
                      shared_sums[i * bands + j]);
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

        if (count[cluster_idx] > 0) {
            centroids[tid] = temp_centroids[tid] / count[cluster_idx];
        }
    }
}

// Kernel CUDA para calcular el error total.
__global__ void compute_error_kernel(baseType* data, u_char* clustering,
                                     baseType* centroids, baseType* yi2,
                                     float* total_error, int bands,
                                     int clusters, int slice) {
    extern __shared__ float shared_err[]; // Memoria compartida dinámica
    float* shared_centroids = shared_err; // Para almacenar errores
    baseType* shared_errors =
        (baseType*)&shared_err[clusters * bands]; // Centroides

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Cargar centroides en memoria compartida de forma colaborativa
    for (int i = local_tid; i < clusters * bands; i += blockDim.x) {
        shared_centroids[i] = centroids[i];
    }
    __syncthreads();

    // Inicializar errores en memoria compartida
    shared_errors[local_tid] = 0.0;

    if (tid < slice) {
        int cluster_idx = clustering[tid];

        baseType xiyi = 0;
        baseType xi2 = 0;

        for (int j = 0; j < bands; j++) {
            xiyi += data[tid * bands + j] *
                    shared_centroids[cluster_idx * bands + j];
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
        atomicAdd(total_error, shared_errors[0]);
    }
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
    double init_start = get_time();

    baseType* centroids = init_centroids(data, clusters);

    execution_data->kmeans_init = get_time() - init_start;

#ifdef VERBOSE
    printf("Centroids initialized on CPU in %.6f seconds\n",
           execution_data->kmeans_init);
#endif

    // Variables en host
    baseType* d_data = NULL;
    baseType* d_centroids = NULL;
    baseType* d_temp_centroids = NULL;
    baseType* d_yi2 = NULL;
    int* d_count = NULL;
    u_char* d_clustering = NULL;
    float* d_total_error = NULL;
    u_char* h_clustering = NULL;

    double malloc_start = get_time();

    // Reserva de memoria en el host para el resultado final
    h_clustering = (u_char*)malloc(data.slice * sizeof(u_char));
    if (!h_clustering) {
        printf("Error: Unable to allocate memory | clustering \n");
        exit(EXIT_FAILURE);
    }

    // Calcular dimensiones de grid y bloques
    int num_blocks_pixels = (data.slice + execution_data->block_size - 1) /
                            execution_data->block_size;
    int num_blocks_clusters = (clusters + execution_data->block_size - 1) /
                              execution_data->block_size;
    int num_blocks_centroids =
        (clusters * data.bands + execution_data->block_size - 1) /
        execution_data->block_size;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

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
    CUDA_CHECK(cudaMalloc((void**)&d_total_error, sizeof(float)));

    execution_data->kmeans_malloc = get_time() - malloc_start;

#ifdef VERBOSE
    printf("Memory allocated in %.6f seconds\n", execution_data->kmeans_malloc);
#endif

    double transfer_start = get_time();

    // Transferir datos a la GPU
    CUDA_CHECK(cudaMemcpy(d_data, data.x,
                          data.slice * data.bands * sizeof(baseType),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids,
                          clusters * data.bands * sizeof(baseType),
                          cudaMemcpyHostToDevice));

    execution_data->kmeans_transfer = get_time() - transfer_start;

#ifdef VERBOSE
    printf("Data transferred to GPU in %.6f seconds\n",
           execution_data->kmeans_transfer);
#endif

    // Calcular yi2 (norma al cuadrado de cada centroide)
    get_yi2_kernel<<<num_blocks_clusters, execution_data->block_size>>>(
        d_yi2, d_centroids, data.bands, clusters);
    CUDA_CHECK(cudaDeviceSynchronize());

    execution_data->kmeans_init = get_time() - init_start;

#ifdef VERBOSE
    printf("Centroids initialized in %.6f seconds\n",
           execution_data->kmeans_init);
#endif

    double execution_start = get_time();

    int shared_mem_size_distance =
        sizeof(baseType) * (clusters * data.bands + clusters);

    int shared_mem_size_sum =
        sizeof(baseType) * clusters * data.bands + sizeof(int) * clusters;

    int shared_mem_size_err =
        sizeof(float) * execution_data->block_size +
        sizeof(baseType) * data.bands * execution_data->clusters;

    // Bucle principal de k-means
    for (int iter = 0; iter < iterations; iter++) {

#ifdef VERBOSE
        double iter_time = get_time();
        printf("Starting iteration %d/%d\n", iter + 1, iterations);
#endif
        double dist_start = get_time();

        assign_clusters_kernel<<<num_blocks_pixels, execution_data->block_size,
                                 shared_mem_size_distance>>>(
            d_data, d_centroids, d_yi2, d_clustering, data.bands, clusters,
            data.slice);
        CUDA_CHECK(cudaDeviceSynchronize());

        execution_data->kmeans_distance += get_time() - dist_start;

        // Reiniciar contadores y sumas para actualizar centroides
        CUDA_CHECK(cudaMemset(d_count, 0, clusters * sizeof(int)));

        // En la última iteración, calcular error total
        if (iter == iterations - 1) {

            double yi2_start = get_time();

            get_yi2_kernel<<<num_blocks_clusters, execution_data->block_size>>>(
                d_yi2, d_centroids, data.bands, clusters);
            CUDA_CHECK(cudaDeviceSynchronize());

            execution_data->kmeans_yi2 += get_time() - yi2_start;

            // Luego calcular error total sin actualizar centroides
            double error_start = get_time();

            CUDA_CHECK(cudaMemset(d_total_error, 0, sizeof(float)));
            compute_error_kernel<<<num_blocks_pixels,
                                   execution_data->block_size,
                                   shared_mem_size_err>>>(
                d_data, d_clustering, d_centroids, d_yi2, d_total_error,
                data.bands, clusters, data.slice);
            CUDA_CHECK(cudaDeviceSynchronize());

            execution_data->kmeans_error += get_time() - error_start;

            // Transferir resultado de clustering al host
            CUDA_CHECK(cudaMemcpy(h_clustering, d_clustering,
                                  data.slice * sizeof(u_char),
                                  cudaMemcpyDeviceToHost));

#ifdef VERBOSE
            printf("Iteration %d completed | Current error: %.6f\n", iter + 1,
                   execution_data->kmeans_error);
#endif
            break;
        }

        double update_start_1 = get_time();

        CUDA_CHECK(cudaMemset(d_temp_centroids, 0,
                              clusters * data.bands * sizeof(baseType)));

        // Sumar puntos por cluster para actualizar centroides
        sum_clusters_kernel<<<num_blocks_pixels, execution_data->block_size,
                              shared_mem_size_sum>>>(
            d_data, d_clustering, d_temp_centroids, d_count, data.bands,
            clusters, data.slice);
        CUDA_CHECK(cudaDeviceSynchronize());

        execution_data->kmeans_update_1 += get_time() - update_start_1;

        double update_start_2 = get_time();

        // Actualizar centroides
        update_centroids_kernel<<<num_blocks_centroids,
                                  execution_data->block_size>>>(
            d_centroids, d_temp_centroids, d_count, data.bands, clusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        execution_data->kmeans_update_2 += get_time() - update_start_2;

        double yi2_start = get_time();

        // Recalcular yi2
        get_yi2_kernel<<<num_blocks_clusters, execution_data->block_size>>>(
            d_yi2, d_centroids, data.bands, clusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        execution_data->kmeans_yi2 += get_time() - yi2_start;

        // Luego calcular error total
        double error_start = get_time();

        CUDA_CHECK(cudaMemset(d_total_error, 0, sizeof(float)));
        compute_error_kernel<<<num_blocks_pixels, execution_data->block_size,
                               shared_mem_size_err>>>(
            d_data, d_clustering, d_centroids, d_yi2, d_total_error, data.bands,
            clusters, data.slice);
        CUDA_CHECK(cudaDeviceSynchronize());

        execution_data->kmeans_error += get_time() - error_start;
    }

    execution_data->kmeans_execution = get_time() - execution_start;

#ifdef VERBOSE
    printf("K-means execution completed in %.6f seconds\n",
           execution_data->kmeans_execution);
#endif

    // Calcular el número máximo de bloques activos por multiprocesador para
    // cada kernel
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &execution_data->maxBlocksPerSM_distance, assign_clusters_kernel,
        execution_data->block_size, shared_mem_size_distance));

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &execution_data->maxBlocksPerSM_sum, sum_clusters_kernel,
        execution_data->block_size, shared_mem_size_sum));

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &execution_data->maxBlocksPerSM_update, update_centroids_kernel,
        execution_data->block_size,
        0)); // Sin memoria compartida dinámica

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &execution_data->maxBlocksPerSM_yi2, get_yi2_kernel,
        execution_data->block_size,
        0)); // Sin memoria compartida dinámica

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &execution_data->maxBlocksPerSM_error, compute_error_kernel,
        execution_data->block_size,
        shared_mem_size_err)); // Memoria compartida para reducción

    execution_data->occupancy_distance =
        (float)(execution_data->maxBlocksPerSM_distance *
                execution_data->block_size) /
        prop.maxThreadsPerMultiProcessor;

    execution_data->occupancy_sum = (float)(execution_data->maxBlocksPerSM_sum *
                                            execution_data->block_size) /
                                    prop.maxThreadsPerMultiProcessor;

    execution_data->occupancy_update =
        (float)(execution_data->maxBlocksPerSM_update *
                execution_data->block_size) /
        prop.maxThreadsPerMultiProcessor;

    execution_data->occupancy_yi2 = (float)(execution_data->maxBlocksPerSM_yi2 *
                                            execution_data->block_size) /
                                    prop.maxThreadsPerMultiProcessor;

    execution_data->occupancy_error =
        (float)(execution_data->maxBlocksPerSM_error *
                execution_data->block_size) /
        prop.maxThreadsPerMultiProcessor;

    double free_start = get_time();

    // Liberar memoria en el dispositivo
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_temp_centroids);
    cudaFree(d_yi2);
    cudaFree(d_count);
    cudaFree(d_clustering);
    cudaFree(d_total_error);

    free(centroids);

    execution_data->kmeans_free = get_time() - free_start;

#ifdef VERBOSE
    printf("Memory freed in %.6f seconds\n", execution_data->kmeans_free);
#endif

    execution_data->kmeans_total = get_time() - kmeans_start;

#ifdef VERBOSE
    printf("K-means total time: %.6f seconds\n", execution_data->kmeans_total);
#endif

    return h_clustering;
}