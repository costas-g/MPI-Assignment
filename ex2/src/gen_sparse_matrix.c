#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "gen_sparse_matrix.h"

int gen_sparse_matrix(int *matrix_out, long long rows, long long cols, float sparsity, int max_val, struct xorshift32_state *state, long long *nnz){
    /* Generate random values and count the number of non-zero elements */
    struct xorshift32_state my_state;
    my_state.a = state->a;

    if (max_val < 1) max_val = RAND_MAX;
    long long total = rows * cols;
    long long nnz_global = 0;
    for (long long i = 0; i < total; i++){
        if ((xorshift32(&my_state) % 10000) >= (size_t)(sparsity*10000)) /* Apply sparsity */ {
            matrix_out[i] = xorshift32(&my_state) % max_val + 1; /* in range [1, max_val] */
            nnz_global++;
        }
    }

    *nnz = nnz_global;
    
    return 0;
}