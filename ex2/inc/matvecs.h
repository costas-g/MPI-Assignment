#ifndef matvecs_h_
#define matvecs_h_

/* Repeated matrix-vector multiplication.
A is the input matrix. Matrix has to be square.
x is the input vector. 
res is the output vector. It should be pre-allocated. 
SIZE is the number of rows or columns of the matrix, and also the number of elements of the vector and each resultant vector. 
ITERS is the number of repeated multiplications. If 0, returns the input vector. */
void matvecs(const int *A, const int *x, int *res, long long size, int iters);

/* Same as matvecs but in parallel */
void matvecs_parallel(const int *A_local, const int *x, int *res, long long size, long long local_rows, int iters, int my_rank, int root_proc);

#endif