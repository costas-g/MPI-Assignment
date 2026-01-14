#ifndef matvecs_csr_h_
#define matvecs_csr_h_

#include "csr_matrix_util.h"

/* Repeated matrix-vector multiplication using CSR sparse matrix representation.
A is the input matrix. Matrix has to be square.
x is the input vector. 
res is the output vector. It should be pre-allocated. 
SIZE is the number of rows or columns of the matrix, and also the number of elements of the vector and each resultant vector. 
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs_csr(csr_matrix_t *A_csr, int *x, int *res, int iters);

/* Same as matvecs but in parallel, with THREAD_COUNT threads. */
void matvecs_csr_parallel(csr_matrix_t *A_csr, int *x, int *res, int iters, int thread_count);

#endif