#ifndef gen_sparse_matrix_h_
#define gen_sparse_matrix_h_

#include "xorshift32.h" /* PRNG by George Marsaglia */

/* Allocates memory for matrix, fills it with random integers with given sparsity, and returns a double pointer to it (pointer to rows).
Also, assigns the input variable NNZ to the number of non-zero elements generated. 
If max_val is less than 2, RAND_MAX is used instead. */
int gen_sparse_matrix(int *matrix_out, long long rows, long long cols, float sparsity, int max_val, int workers, struct xorshift32_state *state, long long *nnz);

#endif
