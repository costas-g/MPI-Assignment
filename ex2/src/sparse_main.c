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

            if (matrix_size % comm_sz != 0) Usage(argv[0], &terminate);                                                 /* matrix size not divisible by number of processes */
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
    double send_time, compute_time, reduce_time, malloc_time;
    double elapsed_time, gen_time;
    double csr_serial_time = 0.;
    double csr_parallel_time;
    double csr_parallel_time_sum;
    double dense_serial_time = 0.;
    double dense_parallel_time;
    double dense_parallel_time_sum;

    /* Object pointers and variables */
    int *matrix_in_p               = NULL;  /* pointer to the input flattened matrix of integers */
    int *matrix_in_dense_partial_p = NULL;  /* pointer to the PARTIAL input flattened matrix of integers */
    // int **matrix_in_pp             = NULL;  /* double pointer to the input matrix of integers (like an array of int pointers) */
    long long nnz;                          /* number of non-zero elements generated */
    int *vec_in_p                  = NULL;  /* pointer to the input vector array of integers */
    csr_matrix_t *matrix_csr_p     = NULL;  /* pointer to the CSR representation of the matrix */
    int *vec_out_csr_ser_p         = NULL;  /* pointer to the output (result) vector for CSR serial */
    int *vec_out_csr_par_p         = NULL;  /* pointer to the output (result) vector for CSR parallel */
    int *vec_out_dense_ser_p       = NULL;  /* pointer to the output (result) vector for DENSE serial */
    int *vec_out_dense_par_p       = NULL;  /* pointer to the output (result) vector for DENSE parallel */
    int *vec_out_csr_partial_p     = NULL;  /* pointer to the PARTIAL output (result) vector for CSR parallel */
    int *vec_out_dense_partial_p   = NULL;  /* pointer to the PARTIAL output (result) vector for DENSE parallel */


    /* =================== Allocate memory for SERIAL =================== */
    if (my_rank == 0) {
        long long total = rows * cols;
        matrix_in_p  = calloc((size_t) (total), sizeof(int));
        matrix_csr_p = malloc(sizeof(csr_matrix_t));

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
            gen_sparse_matrix(matrix_in_p, rows, cols, sparsity, 10, 1, &prng_state, &nnz);
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

        // print_csr_matrix(matrix_csr_p, nnz);
    }


    /* ============================================================================================== */
    /* ========================================== PARALLEL ========================================== */
    /* ============================================================================================== */

    /* -------------------- This is the SEND DATA required for DENSE matvec -------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    start_all = MPI_Wtime();
    int root_proc = 0; /* process 0 sends and gathers */

    /* First, broadcast the matrix size */
    MPI_Bcast(&matrix_size, 1, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD);
    rows = cols = matrix_size; /* matrix is square -- rows and columns are equal in size */

    /* ------- Allocate memory for all for CSR PARALLEL ------- */
    if (my_rank != root_proc) vec_in_p = malloc(cols * sizeof(int)); /* it has already been malloc'ed by root proc during initial generation */
    vec_out_csr_par_p     = malloc(rows * sizeof(int));
    vec_out_csr_partial_p = calloc(rows , sizeof(int));
    
    /* Then broadcast input vector */
    MPI_Bcast(vec_in_p, (int) cols, MPI_INT, root_proc, MPI_COMM_WORLD);

    /* Also broadcast num_mults */
    MPI_Bcast(&num_mults, 1, MPI_INT, root_proc, MPI_COMM_WORLD);

    finish = MPI_Wtime();
    elapsed_time = finish - start_all;
    MPI_Reduce(&elapsed_time, &send_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    // /* ==================== Sparse matrix repeated multiplication PARALLEL ====================== */
    // if (my_rank == 0) {
    //     printf("\n================================================");
    //     vec_out_csr_par_p = malloc(rows * sizeof(int));
    //     printf("\nSparse matrix repeated multiplication PARALLEL...\n");
    //     start = MPI_Wtime(); /* start time */
    //         matvecs_csr_parallel(matrix_csr_p, vec_in_p, vec_out_csr_par_p, num_mults, 1);
    //     finish = MPI_Wtime(); /* finish time */
    //     /* elapsed time */
    //     elapsed_time = finish - start;
    //     printf("  Sparse matrix %dx mult Parallel time (s): %9.6f\n", num_mults, elapsed_time);
    //     // print_matrix(mtx_p, rows, cols);
    //     // print_vector(vec, rows);
    //     // print_vector(vec_out_csr_par_p, rows);
    // }
    
    /* ==================== Dense matrix repeated multiplication PARALLEL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nDense matrix repeated multiplication PARALLEL...\n");
    }

    /* ---------- Allocate memory for all for DENSE PARALLEL ---------- */
    vec_out_dense_par_p     = malloc(rows * sizeof(int));
    vec_out_dense_partial_p = calloc(rows , sizeof(int));

    long long local_rows = rows / comm_sz;
    matrix_in_dense_partial_p = malloc(local_rows * cols * sizeof(int));

    /* Then scatter the matrix */
    long long local_matrix_total = local_rows * cols;
    MPI_Scatter(matrix_in_p, (int) local_matrix_total, MPI_INT, matrix_in_dense_partial_p, (int) local_matrix_total, MPI_INT, root_proc, MPI_COMM_WORLD);
    // printf("\nrank %d: after matrix scatter\n", my_rank);
    // print_matrix(matrix_in_dense_partial_p, local_rows, cols);

    start = MPI_Wtime(); /* start time */
        matvecs_parallel(matrix_in_dense_partial_p, vec_in_p, vec_out_dense_par_p, matrix_size, local_rows, num_mults, my_rank, root_proc);
    finish = MPI_Wtime(); /* finish time */
    /* elapsed time */
    elapsed_time = finish - start;
    MPI_Reduce(&elapsed_time, &dense_parallel_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("  Dense matrix %dx mult Parallel time (s): %9.6f\n", num_mults, dense_parallel_time);
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
    }
    

    /* ============================================================================================== */
    /* =========================================== SERIAL =========================================== */
    /* ============================================================================================== */

    /* ==================== Dense matrix repeated multiplication SERIAL ====================== */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nDense matrix repeated multiplication SERIAL...\n");
        start = MPI_Wtime(); /* start time */
            matvecs(matrix_in_p, vec_in_p, vec_out_dense_ser_p, matrix_size, num_mults);
        finish = MPI_Wtime(); /* finish time */
        /* elapsed time */
        elapsed_time = finish - start;
        printf("  Dense matrix %dx mult Serial time (s):   %9.6f\n", num_mults, elapsed_time);
        dense_serial_time = elapsed_time;

        // print_matrix(matrix_in_p, rows, cols);
        // print_vector(vec_in_p, rows);
        // print_vector(vec_out_dense_ser_p, rows);
        // print_vector(vec_out_dense_par_p, rows);

        /* Compare the two resulting vectors */
        printf("\nComparing Serial with Parallel results...\n");
        long long nerrors = vectors_diffs(vec_out_dense_ser_p, vec_out_dense_par_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
    }

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

        /* Compare the two resulting vectors */
        printf("\nComparing Serial with Parallel results...\n");
        long long nerrors = vectors_diffs(vec_out_csr_ser_p, vec_out_csr_par_p, matrix_size);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
    }

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
    }
    

    /* =============================== Cleanup =============================== */
    /* Free allocated memory */
    // free(matrix_in_pp[0]); // frees the contiguous data block
    // free(matrix_in_pp);
    free(matrix_in_p);
    free(vec_in_p);
    free(vec_out_csr_ser_p);
    free(vec_out_csr_par_p);
    free_csr_matrix(matrix_csr_p);

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