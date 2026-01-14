#ifndef csr_matrix_util_h_
#define csr_matrix_util_h_

#include <stddef.h> /* defines size_t */

/* Struct that holds the pointers to the arrays of the CSR sparse matrix representation. 
 * Also holds the number of rows in the rows field. 
 * Fields: rows, values, col_index, row_ptr. 
 */
struct csr_matrix { 
    long long rows;
    int *values;
    long long *col_index;
    long long *row_ptr;
};

typedef struct csr_matrix csr_matrix_t;

/* Initializes the fields of the csr_matrix object pointed to by the input pointer. Value fields are set to 0, and pointer fields to NULL. */
int init_csr_matrix(csr_matrix_t* csr_matrix);

/* Sets the fields of the csr_matrix object pointed to by the input pointer to the values and pointers passed. */
int set_csr_matrix(csr_matrix_t *csr_matrix_p, long long rows, int* values, long long* col_index, long long* row_ptr);

/* Buils the CSR sparse matrix representation of the input matrix. NNZ required. */
int build_csr_matrix(const int *input_mtx, csr_matrix_t *output_mtx_csr, long long rows, long long cols, long long nnz);

/* Buils the CSR sparse matrix representation of the input matrix in parallel. NNZ required. */
int build_csr_matrix_parallel(const int *input_mtx, csr_matrix_t *output_mtx_csr, long long rows, long long cols, long long nnz, size_t thread_count);

/* Frees the pointers associated with the sparse_matrix_csr struct. */
void free_csr_matrix(csr_matrix_t *mtx_csr);

/* Counts and returns the number of non-zero elements of a matrix. */
long long count_nnz(int **mtx, long long rows, long long cols);

/* Compares two sparse_matrix_csr structs. Returns 1 if they are the same, 0 if not. */
int compare_csr_matrix(csr_matrix_t *A, csr_matrix_t *B, long long nnz);

/* Print the CSR sparse matrix representation arrays. */
void print_csr_matrix(csr_matrix_t *M, long long nnz);

#endif