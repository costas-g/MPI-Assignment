#include <stdio.h>
#include <unistd.h> /* for gethostname */
#include <stdlib.h>
#include <time.h>
#include <mpi.h>    /* For MPI functions, etc */ 

#include "gen_int_array.h"
#include "gen_sparse_matrix.h"
#include "csr_matrix_util.h"
#include "matvecs.h"
#include "matvecs_csr.h"
#include "util_matvec.h"

void mpi_debug_point(int my_rank, int size, int *debug)
{
    for (int r = 0; r < size; r++) {
        if (my_rank == r) {
            printf("[%d] @ %d\n", my_rank, (*debug)++);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void Usage(char* prog_name, int* terminate);

int main(int argc, char* argv[]) {
    int comm_sz;        /* Number of processes  */
    int my_rank;        /* My process rank      */
    int terminate = 0;  /* For terminating all processes if needed */

    long long matrix_size    = 0;   /* row/columnn size of square matrix */
    float sparsity           = 0.;  /* percentage of zero-elements of the matrix */
    int num_mults            = 0;   /* number of repeated multiplications */
    long long rows = 0, cols = 0;   /* rows and columnns of square matrix */


    /* =============== MPI Initialization =============== */
    /* No MPI calls before this */
    /* Start up MPI */
    MPI_Init(&argc, &argv); 

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    
    /* Get name of host machine */
    char hostname[512];
    gethostname(hostname, 512);


    /* ============================= Input ============================= */
    if (my_rank == 0) {
        /* Parse inputs and error check */
        if (argc < 4) 
            Usage(argv[0], &terminate);
        else {
            matrix_size = strtoll(argv[1], NULL, 10); if (matrix_size <= 0)                 Usage(argv[0], &terminate); /* negative size input */
            sparsity    =  strtof(argv[2], NULL);     if (sparsity < 0 || sparsity >= 1)    Usage(argv[0], &terminate); /* invalid percentage input */
            num_mults   =  strtol(argv[3], NULL, 10); if (num_mults < 0)                    Usage(argv[0], &terminate); /* negative number of multiplications input */

            if (matrix_size % comm_sz != 0) Usage(argv[0], &terminate); /* matrix size not divisible by number of processes */
        }
        rows = cols = matrix_size; /* matrix is square -- rows and columns are equal in size */
    }
        
    /* process 0 broadcasts the terminate value to all ranks */
    MPI_Bcast(&terminate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (terminate) {
        MPI_Finalize(); // all ranks finalize cleanly
        exit(0);
    }


    /* =================== Common variables =================== */
    /* Timing variables */
    double start, finish;
    double start_all, finish_all;
    double build_time = 0., send_time, malloc_time;
    double elapsed_time, gen_time;
    double csr_serial_time = 0.;
    double csr_parallel_time;
    double csr_parallel_time_all;
    double csr_parallel_time_sum;
    double dense_serial_time = 0.;
    double dense_parallel_time;
    double dense_parallel_time_all;
    // double dense_parallel_time_sum;

    /* Object pointers and variables */
    int *matrix_in_p                 = NULL;    /* pointer to the input flattened matrix of integers         */
    int *matrix_in_dense_partial_p   = NULL;    /* pointer to the PARTIAL input flattened matrix of integers */
    long long nnz;                              /* number of non-zero elements generated                     */
    int *sendcounts                  = NULL;    /* need to be int[] for Scatterv argument                    */
    int *displs                      = NULL;    /* need to be int[] for Scatterv argument                    */
    int local_nnz;                              /* number of local NNZ (in local matrix rows)                */
    int *vec_in_p                    = NULL;    /* pointer to the input vector array of integers             */
    csr_matrix_t *matrix_csr_p       = NULL;    /* pointer to the CSR representation of the matrix           */
    csr_matrix_t *matrix_csr_local_p = NULL;    /* pointer to the CSR representation of the matrix           */
    long long csr_local_rows;                   /* local rows per process                                    */
    int *csr_local_values            = NULL;    /* pointer to local values csr array                         */
    long long *csr_local_col_index   = NULL;    /* pointer to local col_index csr array                      */
    long long *csr_local_row_ptr     = NULL;    /* pointer to local row_ptr csr array                        */
    long long *csr_extended_row_ptr  = NULL;    /* pointer to global extended row_ptr csr array for scatter  */
    int *vec_out_csr_ser_p           = NULL;    /* pointer to the output (result) vector for CSR serial      */
    int *vec_out_csr_par_p           = NULL;    /* pointer to the output (result) vector for CSR parallel    */
    int *vec_out_dense_ser_p         = NULL;    /* pointer to the output (result) vector for DENSE serial    */
    int *vec_out_dense_par_p         = NULL;    /* pointer to the output (result) vector for DENSE parallel  */


    /* =================== Allocate memory for SERIAL =================== */
    if (my_rank == 0) {
        long long total = rows * cols;
        matrix_in_p         = calloc((size_t) (total), sizeof(int));
        matrix_csr_p        = malloc(sizeof(csr_matrix_t));
        
        vec_in_p            = malloc(cols * sizeof(int));
        vec_out_csr_ser_p   = malloc(rows * sizeof(int));
        vec_out_dense_ser_p = malloc(rows * sizeof(int));
    }


    /* ==================== Generate the matrix and the array of integers (SERIAL ONLY) ====================== */
    if (my_rank == 0) {
        printf("Square Matrix of dimensions NxN with N=%lld, sparsity=%f\nRepeated multiplications: %d\nProcess count: %d\n", matrix_size, sparsity, num_mults, comm_sz);
        printf("\n================================================");
        
        /* Seed the random number generators */
        srand((unsigned int) time(NULL)); /* seed random generator */
        struct xorshift32_state prng_state;
        prng_state.a = (unsigned int) time(NULL); /* seed the PRNG */

        printf("\nGenerating the square matrix of integers...\n");
        start = MPI_Wtime(); /* start time */
            gen_sparse_matrix(matrix_in_p, rows, cols, sparsity, 10, &prng_state, &nnz);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        gen_time = finish - start; 
        printf("  Matrix generation time (s): %9.6f\n", gen_time);
        printf("  NNZ generated: %lld\n", nnz);

        printf("\nGenerating the vector array of integers...\n");
        start = MPI_Wtime(); /* start time */
            gen_int_array(vec_in_p, cols, 10);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        gen_time = finish - start;
        printf("  Vector generation time (s): %9.6f\n", gen_time);

        // print_matrix(matrix_in_p, rows, cols);
        // print_vector(vec_in_p, rows);
    }


    /* ============================= Build CSR Representation (SERIAL ONLY) ============================= */
    MPI_Barrier(MPI_COMM_WORLD);
    start_all = MPI_Wtime();
    if (my_rank == 0) {
        printf("\n================================================");
        
        init_csr_matrix(matrix_csr_p);

        /* Serial CSR Build */ 
        printf("\nSerial CSR build...\n");
        start = MPI_Wtime(); /* start time */
            build_csr_matrix(matrix_in_p, matrix_csr_p, rows, cols, nnz);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        elapsed_time = finish - start;
        printf("  Serial CSR build time (s):   %9.6f\n", elapsed_time);
        build_time = elapsed_time;

        // print_csr_matrix(matrix_csr_p, nnz);
    }

    /* ============================================================================================== */
    /* ========================================== PARALLEL ========================================== */
    /* ============================================================================================== */

    /* -------------------- This is the SEND DATA required for CSR matvec -------------------- */
    int root_proc = 0; /* process 0 sends and gathers */

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    /* First, broadcast the matrix size */
    MPI_Bcast(&matrix_size, 1, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD);
    rows = cols = matrix_size; /* matrix is square -- rows and columns are equal in size */
    
    /* Allocate the input vector in each process */
    if (my_rank != root_proc) vec_in_p = malloc(cols * sizeof(int)); /* it has already been malloc'ed by root proc during initial generation */
    /* Then broadcast input vector */
    MPI_Bcast(vec_in_p, (int) cols, MPI_INT, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* Also broadcast num_mults */
    MPI_Bcast(&num_mults, 1, MPI_INT, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* ------------------------------ Scatter the CSR matrix ------------------------------ */
    csr_local_rows = rows / comm_sz;    /* rows per process */

    /* rank 0 calculates the non-uniform sendcounts and displs to prepare for Scatterv */
    if (my_rank == 0) {
        sendcounts = malloc(comm_sz * sizeof(int));
        displs     = malloc(comm_sz * sizeof(int));
        displs[0]  = 0;
        for (int rank = 0; rank < comm_sz; rank++) {
            sendcounts[rank] = (int) matrix_csr_p->row_ptr[(rank + 1) * csr_local_rows] - (int) matrix_csr_p->row_ptr[rank * csr_local_rows];
            if (rank > 0) displs[rank] = displs[rank - 1] + sendcounts[rank - 1];
        }
    }
    // MPI_Barrier(MPI_COMM_WORLD);

    /* send each local_nnz to each process */
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_nnz, 1, MPI_INT, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* local values array */
    csr_local_values = malloc((size_t) local_nnz * sizeof(int));
    /* Scatterv global values array to local values arrays */
    int *send_values = (my_rank == root_proc) ? matrix_csr_p->values : NULL; /* to avoid dereferencing a NULL in the next line */
    MPI_Scatterv(send_values, sendcounts, displs, MPI_INT, csr_local_values, local_nnz, MPI_INT, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* local col_index array */
    csr_local_col_index = malloc(local_nnz * sizeof(long long));
    /* Scatterv global col_index array to local col_index arrays */
    long long *send_col_index = (my_rank == root_proc) ? matrix_csr_p->col_index : NULL; /* to avoid dereferencing a NULL in the next line */
    MPI_Scatterv(send_col_index, sendcounts, displs, MPI_LONG_LONG, csr_local_col_index, local_nnz, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* rank 0 computes the extended global row_ptr to prepare a uniform non-overlapping scatter */
    if (my_rank == 0) {
        csr_extended_row_ptr = malloc((comm_sz * (csr_local_rows + 1)) * sizeof(long long));
        for (int rank = 0; rank < comm_sz; rank++) {
            for (int row = 0; row < csr_local_rows + 1; row++) {
                csr_extended_row_ptr[rank * (csr_local_rows + 1) + row] = matrix_csr_p->row_ptr[rank * csr_local_rows + row];
            }
        }
    }

    /* local row_ptr array */
    csr_local_row_ptr = malloc((csr_local_rows + 1) * sizeof(long long));
    /* Scatter global row_ptr array to local row_ptr arrays */
    MPI_Scatter(csr_extended_row_ptr, csr_local_rows + 1, MPI_LONG_LONG, csr_local_row_ptr, csr_local_rows + 1, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD); /* implicit barrier */

    /* set the new local csr_matrix representation */
    matrix_csr_local_p  = malloc(sizeof(csr_matrix_t));
    set_csr_matrix(matrix_csr_local_p, csr_local_rows, csr_local_values, csr_local_col_index, csr_local_row_ptr);
    /* ------ FINISHED scatter of CSR matrix ------ */
    /* -------------- FINISHED SEND --------------- */
    finish = MPI_Wtime();
    elapsed_time = finish - start;
    MPI_Reduce(&elapsed_time, &send_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("  Data send time (s): %9.6f\n", send_time);
    }

    
    /* ==================== Sparse matrix repeated multiplication PARALLEL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nSparse matrix repeated multiplication PARALLEL...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime(); /* start time */
    /* Allocate the output result from csr parallel for each process.
     * We need the full vector due to repeated multiplications */
    vec_out_csr_par_p = malloc(rows * sizeof(int));
    finish = MPI_Wtime(); /* finish time */
    /* elapsed time */
    elapsed_time = finish - start;
    MPI_Reduce(&elapsed_time, &malloc_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime(); /* start time */
        matvecs_csr_parallel(matrix_csr_local_p, vec_in_p, vec_out_csr_par_p, rows, num_mults);
    finish_all = MPI_Wtime(); /* finish time */
    /* elapsed time */
    elapsed_time = finish_all - start;
    MPI_Reduce(&elapsed_time, &csr_parallel_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    elapsed_time = finish_all - start_all;
    MPI_Reduce(&elapsed_time, &csr_parallel_time_all, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        csr_parallel_time_sum = build_time + send_time + malloc_time + csr_parallel_time;
        printf("  Sparse matrix %dx mult Parallel time (s): %9.6f\n", num_mults, csr_parallel_time_all);
        printf("                                   clean sum: %9.6f (%5.2f%%)\n",   csr_parallel_time_sum, 100 * csr_parallel_time_sum / csr_parallel_time_all);
        printf("                                     build  :   %9.6f (%5.2f%%)\n",            build_time, 100 *            build_time / csr_parallel_time_sum);
        printf("                                     send   :   %9.6f (%5.2f%%)\n",             send_time, 100 *             send_time / csr_parallel_time_sum);
        printf("                                     malloc :   %9.6f (%5.2f%%)\n",           malloc_time, 100 *           malloc_time / csr_parallel_time_sum);
        printf("                                     compute:   %9.6f (%5.2f%%)\n",     csr_parallel_time, 100 *     csr_parallel_time / csr_parallel_time_sum);
        // print_matrix(mtx_p, rows, cols);
        // print_vector(vec, rows);
        // print_vector(vec_out_csr_par_p, rows);
    }


    /* ==================== Dense matrix repeated multiplication PARALLEL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nDense matrix repeated multiplication PARALLEL...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_all = MPI_Wtime();
    /* ---------- Allocate memory for all for DENSE PARALLEL ---------- */
    vec_out_dense_par_p = malloc(rows * sizeof(int));

    long long local_rows = rows / comm_sz;
    matrix_in_dense_partial_p = malloc(local_rows * cols * sizeof(int));

    /* Then scatter the matrix */
    long long local_matrix_total = local_rows * cols;
    MPI_Scatter(matrix_in_p, (int) local_matrix_total, MPI_INT, matrix_in_dense_partial_p, (int) local_matrix_total, MPI_INT, root_proc, MPI_COMM_WORLD);

    start = MPI_Wtime(); /* start time */
        matvecs_parallel(matrix_in_dense_partial_p, vec_in_p, vec_out_dense_par_p, matrix_size, local_rows, num_mults);
    finish_all = MPI_Wtime(); /* finish time */
    /* elapsed time */
    elapsed_time = finish_all - start;
    MPI_Reduce(&elapsed_time, &dense_parallel_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    elapsed_time = finish_all - start_all;
    MPI_Reduce(&elapsed_time, &dense_parallel_time_all, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("  Dense matrix %dx mult Parallel time (s): %9.6f\n", num_mults, dense_parallel_time_all);
        printf("                                    compute: %9.6f (%5.2f%%)\n", dense_parallel_time, 100 * dense_parallel_time / dense_parallel_time_all);
        // print_matrix(matrix_in_p, rows, cols);
        // print_vector(vec_in_p, rows);
        // print_vector(vec_out_dense_par_p, rows);
    }

    /* =============================== Compare Dense vs CSR PARALLEL ============================ */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nFINAL: Comparing Dense vs Sparse matrix (parallel) multiplication results...\n");
        long long nerrors = vectors_diffs(vec_out_dense_par_p, vec_out_csr_par_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
        printf("  Dense / CSR Speedup in Parallel: %5.3f\n", dense_parallel_time_all/csr_parallel_time_all);
    }
    

    /* ============================================================================================== */
    /* =========================================== SERIAL =========================================== */
    /* ============================================================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    /* ==================== Dense matrix repeated multiplication SERIAL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nDense matrix repeated multiplication SERIAL...\n");
        start = MPI_Wtime(); /* start time */
            matvecs(matrix_in_p, vec_in_p, vec_out_dense_ser_p, matrix_size, num_mults);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        elapsed_time = finish - start;
        dense_serial_time = elapsed_time;
        printf("  Dense matrix %dx mult Serial time (s):   %9.6f\n", num_mults, elapsed_time);
        

        // print_matrix(matrix_in_p, rows, cols);
        // print_vector(vec_in_p, rows);
        // print_vector(vec_out_dense_ser_p, rows);
        // print_vector(vec_out_dense_par_p, rows);

        /* Compare the DENSE Serial with Parallel result vectors */
        printf("\nComparing Serial with Parallel results...\n");
        long long nerrors = vectors_diffs(vec_out_dense_ser_p, vec_out_dense_par_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
        printf("  Dense MV Speedup: %5.3f\n", dense_serial_time/dense_parallel_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* ==================== Sparse matrix repeated multiplication SERIAL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nSparse matrix repeated multiplication SERIAL...\n");
        start = MPI_Wtime(); /* start time */
            matvecs_csr(matrix_csr_p, vec_in_p, vec_out_csr_ser_p, num_mults);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        elapsed_time = finish - start;
        printf("  Sparse matrix %dx mult Serial time (s):   %9.6f\n", num_mults, elapsed_time);
        csr_serial_time = elapsed_time;

        // print_matrix(matrix_in_p, rows, cols);
        // print_vector(vec_in_p, rows);
        // print_vector(vec_out_csr_ser_p, rows);
        // print_vector(vec_out_csr_par_p, rows);

        /* Compare the CSR Serial with Parallel result vectors */
        printf("\nComparing Serial with Parallel results...\n");
        long long nerrors = vectors_diffs(vec_out_csr_ser_p, vec_out_csr_par_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
        printf("\n  CSR MV Speedup: %5.3f\n", csr_serial_time/csr_parallel_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* =============================== Compare Dense vs CSR SERIAL ============================ */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nFINAL: Comparing Dense vs Sparse matrix (serial) multiplication results...\n");
        long long nerrors = vectors_diffs(vec_out_dense_ser_p, vec_out_csr_ser_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
        printf("  Dense / CSR Speedup in Serial: %5.3f\n", dense_serial_time/csr_serial_time);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    /* =============================== Cleanup =============================== */
    /* Free allocated memory */
    if (my_rank == 0) {
        free(csr_extended_row_ptr);
        free(displs);
        free(sendcounts);
        free(vec_out_dense_ser_p);
        free(vec_out_csr_ser_p);
        free(matrix_csr_p);
        free(matrix_in_p);
    }

    free(matrix_in_dense_partial_p);
    free(vec_out_dense_par_p);
    free(matrix_csr_local_p);
    free(csr_local_row_ptr);
    free(csr_local_col_index);
    free(csr_local_values);
    free(vec_out_csr_par_p);
    free(vec_in_p);


    /* Shut down MPI */
    MPI_Finalize();
    /* No MPI calls after this */

    return 0;
} /* main */

/*--------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message indicating how program should be started
 *            and terminate.
 */
void Usage(char *prog_name, int *terminate) {
    if (terminate != NULL)
        if (*terminate) return;

    fprintf(stderr, "Usage: %s <matrix_size> <sparsity> <num_mults>\n", prog_name);
    fprintf(stderr, "   matrix_size: Row/column size (square matrix). Must be positive. Must be divisible by the number of processes.\n");
    fprintf(stderr, "   sparsity: Percentage of zero-elements. Should be a float from 0 to 1.\n");
    fprintf(stderr, "   num_mults: Number of repeated multiplications. Should be non-negative.\n");

    /* trigger exit */
    if (terminate != NULL) (*terminate)++;
    return;
} /* Usage */