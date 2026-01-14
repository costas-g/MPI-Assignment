#ifndef matvecs_h_
#define matvecs_h_

/* Repeated matrix-vector multiplication.
A is the input matrix. Matrix has to be square.
x is the input vector. 
res is the output vector. It should be pre-allocated. 
SIZE is the number of rows or columns of the matrix, and also the number of elements of the vector and each resultant vector. 
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs(const int *A, const int *x, int *res, long long size, int iters);

/* Repeated matrix-vector multiplication in parallel MPI.
A_local is the partial input matrix, i.e. only the respective rows for each process. Matrix has to be square and rows must be divisible by the number of processes.
x is the full input vector. 
res is the full output vector. It should be pre-allocated. 
MATRIX_ROWS is the number of rows or columns of the matrix, and also the number of elements of the vector and each resultant vector.
LOCAL_ROWS is the number of rows of the matrix associated with each process. It is MATRIX_ROWS/NUM_OF_PROCESSES. 
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs_parallel(const int *A_local, const int *x, int *res, long long matrix_rows, long long local_rows, int iters);

#endif