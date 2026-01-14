#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "matvecs_csr.h"
#include "csr_matrix_util.h"

void matvecs_csr(const csr_matrix_t *A_csr, const int *x, int *res, int iters){
    long long i, j;
    int r;
    long long rows = A_csr->rows;
    long long cols = rows; /* cols = rows for square matrix */

    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (i = 0; i < cols; i++) {
            res[i] = x[i];
        }
        return;
    }

    /* Pointer that can point to two arrays. The two arrays will be used as intermediate result arrays 
     * In each stage/iteration, one is used as input to be read and the other gets written with the result. 
     * At every next stage, they are switched, so that the result array is now read as input and the input array is overwrritten with the new result.  */
    int **x_tmp = malloc(2 * sizeof(int*));
    x_tmp[0] = malloc(2*cols * sizeof(int));
    x_tmp[1] = &x_tmp[0][cols];

    /* Copy input x vector to intermediate x_tmp vector. */
    for (i = 0; i < cols; i++) {
        x_tmp[0][i] = x[i];
    }
    
    int *x_read;
    int *x_write;

    for (r = 0; r < iters; r++) {
        x_read = x_tmp[r % 2];
        x_write = x_tmp[(r + 1) % 2];

        for (i = 0; i < rows; i++) {
            x_write[i] = 0;
            for (j = A_csr->row_ptr[i]; j < A_csr->row_ptr[i+1]; j++) {
                x_write[i] += A_csr->values[j] * x_read[A_csr->col_index[j]];
            }
        }
    }

    /* Copy result to output memory */
    for (i = 0; i < cols; i++) {
        res[i] = x_write[i];
    }

    /* Free allocated memory */
    free(x_tmp[0]);
    free(x_tmp);

    return;
}


void matvecs_csr_parallel(const csr_matrix_t *A_local_csr, const int *x_full, int *res_full, long long rows, int iters) {
    long long local_rows = A_local_csr->rows;
    /* Copy input vector to output vector. */
    for (long long i = 0; i < rows; i++) {
        res_full[i] = x_full[i];
    }
    
    if (iters < 1) return; /* if iterations are 0, quick return the input vector */

    /* fix row_ptr arrays */
    long long base = A_local_csr->row_ptr[0];
    for (long long row = 0; row < local_rows + 1; row++) {
        A_local_csr->row_ptr[row] -= base;
    }

    int *x_write = NULL; /* local, temporary pointer for (over)writing intermediate result */
    x_write = calloc(local_rows , sizeof(int));           /* allocate memory for the array */
    
    int sum;
    for (int r = 0; r < iters; r++) {
        for (long long i = 0; i < local_rows; i++) {
            sum = 0;
            for (long long j = A_local_csr->row_ptr[i]; j < A_local_csr->row_ptr[i+1]; j++) {
                sum += A_local_csr->values[j] * res_full[A_local_csr->col_index[j]];
            }
            x_write[i] = sum;
        } 

        /* Gather distributed x_write vector to replicated res_full vector, ready for next iteration's input or final output */
        MPI_Allgather(x_write, local_rows, MPI_INT, res_full, local_rows, MPI_INT, MPI_COMM_WORLD);
        /* implicit barrier */

        /* need barrier to synchronize next stage -- we have implicit barrier by MPI_Allgather */
    }
    
    /* Free allocated memory */
    free(x_write);

    return;
}