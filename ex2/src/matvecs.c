#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "matvecs.h"
#include "util_matvec.h"

void matvecs(const int *A, const int *x, int *res, long long size, int iters){
    long long i, j;
    int r;

    if (iters < 1) {
        /* Copy input vector to output vector. */
        for (i = 0; i < size; i++) {
            res[i] = x[i];
        }
        return;
    }

    /* Pointer that can point to two arrays. The two arrays will be used as intermediate result arrays 
     * In each stage/iteration, one is used as input to be read and the other gets written with the result. 
     * At every next stage, they are switched, so that the result array is now read as input and the input array is overwrritten with the new result.  */
    int **x_tmp = malloc(2 * sizeof(int*));
    x_tmp[0] = malloc(2*size * sizeof(int)); /* allocate memory for the two arrays and assign them */
    x_tmp[1] = &x_tmp[0][size]; /* assign the address of the second array to the pointer x_tmp_global[1] */

    /* Copy input x vector to intermediate x_tmp vector. */
    for (i = 0; i < size; i++) {
        x_tmp[0][i] = x[i];
    }
    
    int *x_read;
    int *x_write;

    for (r = 0; r < iters; r++) {
        x_read = x_tmp[r % 2];
        x_write = x_tmp[(r + 1) % 2];
        for (i = 0; i < size; i++) {
            x_write[i] = 0;
            for (j = 0; j < size; j++) {
                x_write[i] += A[i*size + j] * x_read[j];
            }
        }
    }

    /* Copy result to output memory */
    for (i = 0; i < size; i++) {
        res[i] = x_write[i];
    }

    /* Free allocated memory */
    free(x_tmp[0]);
    free(x_tmp);

    return;
}


void matvecs_parallel(const int *A_local, const int *x, int *res, long long matrix_rows, long long local_rows, int iters, int my_rank, int root_proc) {
    long long cols = matrix_rows;
    /* Copy input vector to output vector. */
    for (long long i = 0; i < cols; i++) {
        res[i] = x[i];
    }
    if (iters < 1) return; /* if iterations are 0, quick return the input vector */

    int *x_write = NULL; /* local, temporary pointer for (over)writing intermediate result */
    x_write = calloc(local_rows , sizeof(int));           /* allocate memory for the array */

    int sum; 
    for (int r = 0; r < iters; r++) {
        for (long long i = 0; i < local_rows; i++) {
            sum = 0;
            for (long long j = 0; j < cols; j++) {
                sum += A_local[i * cols + j] * res[j];
            }
            x_write[i] = sum;
        }

        /* Gather distributed x_write vector to replicated res vector, ready for next iteration's input or final output */
        MPI_Allgather(x_write, local_rows, MPI_INT, res, local_rows, MPI_INT, MPI_COMM_WORLD);
        /* implicit barrier */

        /* need barrier to synchronize next stage -- we have implicit barrier by MPI_Allgather */
    }

    /* Free allocated memory */
    free(x_write);

    return;
}