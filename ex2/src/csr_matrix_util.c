#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "csr_matrix_util.h"


int init_csr_matrix(csr_matrix_t *csr_matrix_p) {
    if (csr_matrix_p == NULL) return 1;

    csr_matrix_p->rows      = 0; 
    csr_matrix_p->values    = NULL; 
    csr_matrix_p->col_index = NULL; 
    csr_matrix_p->row_ptr   = NULL;

    return 0;
}

int set_csr_matrix(csr_matrix_t *csr_matrix_p, long long rows, int* values, long long* col_index, long long* row_ptr) {
    if (csr_matrix_p == NULL) return 1;

    csr_matrix_p->rows      = rows; 
    csr_matrix_p->values    = values; 
    csr_matrix_p->col_index = col_index; 
    csr_matrix_p->row_ptr   = row_ptr;

    return 0;
}

int build_csr_matrix(const int *input_mtx, csr_matrix_t *output_mtx_csr, long long rows, long long cols, long long nnz){
    csr_matrix_t *csr = output_mtx_csr;
    // long long nnz = count_nnz(input_mtx, rows, cols); /* Use this function to not require nnz input by user. Will take more time though. */
    csr->rows = rows;
    /* Condider doing a single malloc and then assigning each pointer with address offset. Potentially faster. 
     * Will need correct alignment of types. Malloc larger types first (long long: 8 bytes, int: 4 bytes)
     */
    csr->row_ptr   = malloc( (rows+1) * sizeof(long long));
    csr->col_index = malloc( nnz * sizeof(long long));
    csr->values    = malloc( nnz * sizeof(int));

    int val;
    long long idx = 0;
    csr->row_ptr[0] = 0;

    for (long long i = 0; i < rows; i++) {
        csr->row_ptr[i+1] = csr->row_ptr[i];
        for (long long j = 0; j < cols; j++) {
            val = input_mtx[i*cols + j];
            if (val){
                // add check if idx > nnz-1
                csr->values[idx] = val;
                csr->col_index[idx] = j;
                idx++;
                csr->row_ptr[i+1]++;
            }
        }
    }

    if (csr->row_ptr[rows] == nnz){
        // printf("  CSR representation built successfully. NNZ = %lld\n", csr->row_ptr[rows]);
        return 1;
    } else {
        // printf("  ERROR: Mismatch. Given NNZ = %lld. Counted NNZ = %lld\n", nnz, csr->row_ptr[rows]);
        return 0;
    }
};

// int build_csr_matrix_parallel(const int *input_mtx, csr_matrix_t *output_mtx_csr, long long rows, long long cols, long long nnz, size_t thread_count){
//     csr_matrix_t *csr = output_mtx_csr;
//     // long long nnz = count_nnz(input_mtx, rows, cols); /* Use this function to not require nnz input by user. Will take more time though. */
//     csr->rows = rows;
//     /* Condider doing a single malloc and then assigning each pointer with address offset. Potentially faster. 
//      * Will need correct alignment of types. Malloc larger types first (long long: 8 bytes, int: 4 bytes)
//      */
//     csr->row_ptr   = malloc( (rows+1) * sizeof(long long));
//     csr->col_index = malloc( nnz * sizeof(long long));
//     csr->values    = malloc( nnz * sizeof(int));
//     csr->row_ptr[0] = 0;

//     /* Local variables of each thread */
//     int my_val;
//     long long my_col, my_idx = 0, my_row_idx = 0;
//     /* Local arrays of each thread. */
//     int *val_local;
//     long long *col_local, *row_local;
//     long long **row_locals = calloc(thread_count, sizeof(*row_local));
//     long long *our_rows = calloc(thread_count, sizeof(long long));
//     long long *our_nnzs = calloc(thread_count, sizeof(long long));

//     # pragma omp parallel num_threads(thread_count) private(my_val, my_col, val_local, col_local, row_local) firstprivate(my_idx, my_row_idx)
//     {   
//         #ifdef _OPENMP
//         int tid = omp_get_thread_num();
//         int nthreads = omp_get_num_threads();
//         #else
//         int tid = 0;
//         int nthreads = 1;
//         #endif

//         /* We need to know which threads work on which rows because we need to combine them in order at the end. */
//         /* So, calculate and assign the rows specifically for each thread. */
//         long long base_chunk = rows / nthreads;
//         int rem = rows % nthreads;  // remainder rows to spread

//         long long my_start = tid * base_chunk + (tid < rem ? tid : rem);
//         long long my_end   = my_start + base_chunk + (tid < rem ? 1 : 0);
//         long long my_rows = my_end - my_start;

//         // printf("Thread %d works on rows %lld to %lld. Total: %lld rows.\n", tid, my_start, my_end-1, my_rows);

//         /* Each thread allocates its private array */
//         /* Need to allocate nnz elements for the worst case */
//         val_local = calloc(nnz, sizeof(int)); /* Initialize values to 0 so that we may find where each thread's subarray should end. */
//         col_local = malloc(nnz * sizeof(long long));
//         // long long rows_per_thread = (rows+1)/final_thread_count;
//         row_local = malloc((my_rows+1) * sizeof(long long));
//         row_locals[tid] = row_local;
//         row_local[my_row_idx] = 0; /* Initialize first row_ptr element to 0 for each thread. */
//         // long long my_rows = 0; /* Each thread counts the number of rows it has worked on. So we know where each row_local should end. */

//         for (long long i = my_start; i < my_end; i++) {
//             // my_rows++;
//             row_local[my_row_idx+1] = row_local[my_row_idx];
//             for (long long j = 0; j < cols; j++) {
//                 my_val = input_mtx[i*cols + j];
//                 if (my_val){
//                     // add check if idx > nnz-1
//                     val_local[my_idx] = my_val;
//                     col_local[my_idx] = j;
//                     my_idx++;
//                     row_local[my_row_idx+1]++;
//                     // my_nnz++;
//                 }
//             }
//             my_row_idx++;
//         }

//         our_rows[tid] = my_rows;
//         our_nnzs[tid] = row_local[my_rows]; /* last element of row array is always the NNZ */

//         # pragma omp barrier /* barrier */


//         /* -------------------------------- Combine results - in parallel -------------------------------- */

//         /* First fix the values of each row_ptr subarray - in sequence */
//         # pragma omp single
//         {
//             for (int t = 1; t < nthreads; t++) {
//                 for (long long i = 0; i < our_rows[t]+1; i++) {
//                     row_locals[t][i] += row_locals[t-1][our_rows[t-1]];
//                 }
//             }
//         } /* Implicit barrier */

//         /* Then combine the values and col_index subarrays - in parallel */
//         long long my_offset = row_local[0];
//         for (long long i = 0; i < our_nnzs[tid]; i++) {
//             my_val = val_local[i];
//             my_col = col_local[i];
//             if (my_val){
//                 /* It shouldn't be a critical section because each thread should write to different addresses if the method works correctly. */
//                 csr->values    [i + my_offset] = my_val;
//                 csr->col_index [i + my_offset] = my_col;
//             }
//         }

//         /* Finally combine the row_ptr subarrays - in parallel */
//         for (long long i = 1; i < my_rows+1; i++) {
//             /* It shouldn't be a critical section because each thread should write to different addresses if the method works correctly. */
//             csr->row_ptr[i + my_start] = row_local[i];
//         }

//         /* Free no longer needed allocated memory */
//         free(val_local);
//         free(col_local);
//         free(row_local);
//     }

//     /* Free no longer needed allocated memory */
//     free(our_rows);
//     free(our_nnzs);
//     free(row_locals);
    
//     if (csr->row_ptr[rows] == nnz){
//         // printf("  CSR representation built successfully. NNZ = %lld\n", csr->row_ptr[rows]);
//         return 1;
//     } else {
//         // printf("  ERROR: Mismatch. Given NNZ = %lld. Counted NNZ = %lld\n", nnz, csr->row_ptr[rows]);
//         return 0;
//     }
// }

void free_csr_matrix(csr_matrix_t *mtx_csr){
    free(mtx_csr->values);
    free(mtx_csr->col_index);
    free(mtx_csr->row_ptr);
    free(mtx_csr);

    return;
}

long long count_nnz(int **mtx, long long rows, long long cols){
    long long i, j, nnz = 0;
    int val;

    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            val = mtx[i][j];
            if (val) nnz++;
        }
    }
    
    return nnz;
}

int compare_csr_matrix(csr_matrix_t *A, csr_matrix_t *B, long long nnz){
    if (A->rows != B->rows)
        return 0;

    for (long long i = 0; i < nnz; i++){
        if (A->values[i] != B->values[i]){
            return 0;
        }
        if (A->col_index[i] != B->col_index[i]){
            return 0;
        }
    }

    for (long long i = 0; i < A->rows+1; i++){
        if (A->row_ptr[i] != B->row_ptr[i]){
            return 0;
        }
    }

    return 1;
}

void print_csr_matrix(csr_matrix_t *M, long long nnz){
    printf("\nPrinting CSR matrix...\n");
    printf("  NNZ = %lld\n", nnz);

    if (M->values != NULL){
        printf("  values  = [");
        for (long long i = 0; i < nnz; i++){
            printf("%d, ", M->values[i]);
        }
        printf("]\n");
    }
        
    if (M->col_index != NULL){
        printf("  col_ind = [");
        for (long long i = 0; i < nnz; i++){
            printf("%lld, ", M->col_index[i]);
        }
        printf("]\n");
    }

    if (M->row_ptr != NULL){
        printf("  row_ptr = [");
        for (long long i = 0; i < M->rows+1; i++){
            printf("%lld, ", M->row_ptr[i]);
        }
        printf("]\n");
    }
    
    return;
}