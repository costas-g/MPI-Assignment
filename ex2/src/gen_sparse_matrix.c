#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "gen_sparse_matrix.h"

int gen_sparse_matrix(int *matrix_out, long long rows, long long cols, float sparsity, int max_val, int workers, struct xorshift32_state *state, long long *nnz){
    /* Allocation for all the matrix elements, initialized to 0 */
    long long total = rows * cols;
    // int *data = calloc((size_t) (total), sizeof(int));
    // if (!data) {
    //     perror("malloc data");
    //     exit(EXIT_FAILURE);
    // }

    // /* Allocation of ROWS pointers to int pointers (row arrays) */
    // int **mtx = malloc((size_t)(rows) * sizeof(int*));
    // if (!mtx) {
    //     perror("malloc mtx");
    //     exit(EXIT_FAILURE);
    // }

    // /* Assign the row pointers */
    // for (long long i = 0; i < rows; i++){
    //     mtx[i] = &data[i * cols];
    // }
    
    /* Generate random values and count the number of non-zero elements */
    long long nnz_global = 0;
    uint32_t num_threads = workers;
    if (max_val < 1) max_val = RAND_MAX;
    if (workers < 1) num_threads = 1;
    # pragma omp parallel num_threads(num_threads)
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        // int nthreads = omp_get_num_threads();
        #else
        int tid = 0;
        // int nthreads = 1;
        #endif

        struct xorshift32_state my_state;
        my_state.a = state->a + tid;

        // # pragma omp for reduction(+:nnz_global)
        // for (long long i = 0; i < rows; i++){
        //     for (long long j = 0; j < cols; j++){
        //         if ((xorshift32(&my_state) % 10000) >= (size_t)(sparsity*10000)) /* Apply sparsity */ {
        //             mtx[i][j] = xorshift32(&my_state) % max_val + 1; /* in range [1, max_val] */
        //             nnz_global++;
        //         }
        //     }
        // }
        # pragma omp for reduction(+:nnz_global)
        for (long long i = 0; i < total; i++){
            if ((xorshift32(&my_state) % 10000) >= (size_t)(sparsity*10000)) /* Apply sparsity */ {
                matrix_out[i] = xorshift32(&my_state) % max_val + 1; /* in range [1, max_val] */
                nnz_global++;
            }
        }
    }

    *nnz = nnz_global;
    
    return 0;
}