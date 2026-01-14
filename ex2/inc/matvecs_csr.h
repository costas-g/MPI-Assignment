#ifndef matvecs_csr_h_
#define matvecs_csr_h_

#include "csr_matrix_util.h"

/* Repeated matrix-vector multiplication using CSR sparse matrix representation.
A is the input matrix. Matrix has to be square.
x is the input vector. 
res is the output vector. It should be pre-allocated. 
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs_csr(const csr_matrix_t *A_csr, const int *x, int *res, int iters);

/* Repeated matrix-vector multiplication in PARALLEL using CSR sparse matrix representation.
A_local is the local-partial input matrix in CSR form. Matrix has to be square.
x_full is the full input vector. 
res_full is the output vector. It should be pre-allocated. 
MATRIX_ROWS are the rows of the matrix. (Square matrix.)
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs_csr_parallel(const csr_matrix_t *A_local_csr, const int *x_full, int *res_full, long long matrix_rows, int iters);

#endif