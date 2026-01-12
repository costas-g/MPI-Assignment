#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>    /* For MPI functions, etc */ 

#include "poly_gen_full.h"
#include "poly_mult_serial.h"
#include "poly_mult_parallel.h"
#include "poly_util.h"


void Usage(char* prog_name, int* terminate);
// void Build_mpi_type_input(long long* deg_p, long long* poly_A_p, MPI_Datatype* input_mpi_t_p);
// void Bcast_input(int source_rank, long long* deg_p, long long* poly_A_p);

int main(int argc, char* argv[]) {
    int comm_sz;                /* Number of processes  */
    int my_rank;                /* My process rank      */
    int terminate = 0;          /* For terminating all processes if needed */
    size_t deg_input, deg_R; /* polynomial degree    */


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
        long long deg = 0;
        if (argc < 2) 
            Usage(argv[0], &terminate);
        else {
            deg = strtoll(argv[1], NULL, 10); 
            if (deg < 1)            Usage(argv[0], &terminate); /* invalid degree value */
            if ((deg + 1) % comm_sz != 0) Usage(argv[0], &terminate); /* degree value not divisible by number of processes */
        }
        deg_input = (size_t) deg;   /* polynomials have the same degree */
        deg_R = 2 * deg_input;      /* resultant polynomial has double degree */
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
    double elapsed_time, gen_time, serial_time = 0., parallel_time, parallel_time_sum;

    /* polynomial pointers */
    long long *poly_A = NULL, *poly_B = NULL;           /* input polynomials */
    long long *poly_A_part = NULL;              /* partial input polynomial  */
    long long *poly_R_serial = NULL;                    /* resultant polynomial from serial */
    long long *poly_R_part = NULL, *poly_R_full = NULL; /* resultant polynomial from parallel -- partial and full, but both full-sized */
    

    /* ==================== Generate the two polynomials ====================== */
    if (my_rank == 0) {
        printf("Multiplication of two %ld-degree polynomials using %d processes.\n", deg_input, comm_sz);
        printf("\n================================================");
        printf("\nGenerating Polynomials...\n");
        
        srand((unsigned int) time(NULL));   /* seed random generator */
        int max_coeff = 10;                 /* maximum coefficient value (absolute value) */

        start = MPI_Wtime(); /* start time */
            poly_A = poly_gen_full(deg_input, max_coeff);
            poly_B = poly_gen_full(deg_input, max_coeff);
        finish = MPI_Wtime(); /* finish time */

        /* elapsed time */
        gen_time = finish - start; 
        printf("  Polynomials generation time (s): %9.6f\n", gen_time);
    }
    
    
    /* ====================== Serial Poly Multiplication ====================== */
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nSerial Poly Multiplication...\n");

        poly_R_serial = poly_mult_serial(poly_A, deg_input, poly_B, deg_input, &elapsed_time); /* implicit malloc */

        printf("  Serial poly mult time (s):   %9.6f\n", elapsed_time);
        serial_time = elapsed_time;

        #ifdef DEBUG
        print_poly(poly_A, deg_input);
        print_poly(poly_B, deg_input);
        print_poly(poly_R_serial, deg_R);
        #endif
    }
    MPI_Barrier(MPI_COMM_WORLD); /* need this barrier to stop other processes from advancing, interfering with serial performance */


    /* =================================== Parallel Poly Multiplication =================================== */
    if (my_rank == 0) { 
        printf("\n================================================");
        printf("\nParallel Poly Multiplication\n");

        /* Send data to all */
        printf("\nSending data to all processes...\n");
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    start_all = MPI_Wtime();
    /* ------------------- Process 0 sends the input data to all ranks ------------------- */
    int root_proc = 0; /* process 0 sends */

    /* First, broadcast the degree */
    MPI_Bcast(&deg_input, 1, MPI_UNSIGNED_LONG, root_proc, MPI_COMM_WORLD);
    deg_R = 2 * deg_input;  /* resultant polynomial has double degree */

    /* Then allocate memory to all other processes given the degree */
    if (my_rank != 0) poly_B = malloc((deg_input + 1) * sizeof(long long));

    /* Then broadcast poly_B */
    MPI_Bcast(poly_B, (int) deg_input + 1, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD);

    /* Then allocate memory for poly_A_part */
    size_t local_terms = (deg_input + 1) / comm_sz;
    poly_A_part = malloc(local_terms * sizeof(long long));

    /* Then scatter poly_A */
    MPI_Scatter(poly_A, (int) local_terms, MPI_LONG_LONG, poly_A_part, (int) local_terms, MPI_LONG_LONG, root_proc, MPI_COMM_WORLD);

    finish = MPI_Wtime();
    elapsed_time = finish - start_all;
    MPI_Reduce(&elapsed_time, &send_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    /* ---------------------------- Allocate memory for the result ---------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime(); /* start time */
    /* Allocate for final gathered result in process 0 */
    if (my_rank == 0) {
        poly_R_full = calloc(deg_R + 1, sizeof(long long));
        if (!poly_R_full) {
            fprintf(stderr, "Rank %d: calloc poly_R_full failed, deg_R=%ld\n", my_rank, deg_R);
            exit(EXIT_FAILURE);
        }
    }

    /* Allocate for partial result in all processes */
    poly_R_part = calloc(deg_R + 1, sizeof(long long)); 
    if (!poly_R_part) {
        fprintf(stderr, "Rank %d: calloc poly_R_part failed, deg_R=%ld\n", my_rank, deg_R);
        exit(EXIT_FAILURE);
    }

    finish = MPI_Wtime(); /* finish time */
    elapsed_time = finish - start;
    MPI_Reduce(&elapsed_time, &malloc_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);
    
    /* ----------------------------------------- Parallel Polynomial Multiplication ----------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime(); /* start time */

    /* Parallel computation */
    poly_mult_partial(poly_A_part, deg_input, poly_B, local_terms, poly_R_part, (size_t) my_rank); /* implicit malloc */

    finish = MPI_Wtime(); /* finish time */
    elapsed_time = finish - start;
    MPI_Reduce(&elapsed_time, &compute_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime(); /* start time */

    /* Reduce result (adding distributed vectors) */
    MPI_Reduce(poly_R_part, poly_R_full, (int) deg_R + 1, MPI_LONG_LONG, MPI_SUM, root_proc, MPI_COMM_WORLD);

    finish_all = MPI_Wtime(); /* finish time */
    elapsed_time = finish_all - start;
    MPI_Reduce(&elapsed_time, &reduce_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    elapsed_time = finish_all - start_all;
    MPI_Reduce(&elapsed_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);

    if (my_rank == 0) {
        parallel_time_sum = send_time + malloc_time + compute_time + reduce_time;
        printf("  Parallel poly mult time (s): %9.6f\n", parallel_time);
        printf("               actual sum (s): %9.6f\n", parallel_time_sum);
        printf("                      send   :   %9.6f (%4.1f%%)\n", send_time   , send_time    / parallel_time * 100);
        printf("                      malloc :   %9.6f (%4.1f%%)\n", malloc_time , malloc_time  / parallel_time * 100);
        printf("                      compute:   %9.6f (%4.1f%%)\n", compute_time, compute_time / parallel_time * 100);
        printf("                      reduce :   %9.6f (%4.1f%%)\n", reduce_time , reduce_time  / parallel_time * 100);
    }


    /* ------------------ Speedup calculation ------------------ */
    if (my_rank == 0) {
        printf("\nSpeedup: %.3f", serial_time/parallel_time);
        printf("\n w/ sum: %.3f", serial_time/parallel_time_sum);
        printf("\n");
    }
    

    /* ------------------------- Confirm correctness ------------------------- */
    if (my_rank == 0) {
        printf("\n================================================");
        printf("\nComparing Serial & Parallel poly mult results...\n");
        long long nerrors;
        nerrors = poly_count_errors(poly_R_full, poly_R_serial, deg_input);
        if (nerrors == 0) {
            printf("  Results match!\n");
        } else {
            printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
        }
    }
    

    /* ==================================== Cleanup ==================================== */
    /* Free allocated memory */
    if (my_rank == 0) {
        free(poly_R_full);
        free(poly_R_serial);
        free(poly_A);
    }
    
    free(poly_R_part);
    free(poly_A_part);
    free(poly_B);

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
    fprintf(stderr, "Usage: %s <degree>\n", prog_name);
    fprintf(stderr, "   degree: Degree of the polynomials. Must be positive. Degree + 1 must be divisible by the number of processes.\n");

    /* trigger exit */
    if (terminate != NULL) (*terminate)++;
    return;
} /* Usage */

// /*--------------------------------------------------------------------
//  * Function:  Build_mpi_type_input
//  * Purpose:   Create a new derived datatyped for the input to be broadcasted to all processes.
//  */
// void Build_mpi_type_input(long long* deg_p, long long* poly_A_p, MPI_Datatype* input_mpi_t_p) {
//     int array_of_blocklengths[2] = {1, (int) *deg_p + 1};
//     MPI_Datatype array_of_types[2] = {MPI_LONG_LONG, MPI_LONG_LONG};
//     MPI_Aint deg_addr, poly_A_addr;
//     MPI_Aint array_of_displacements[2] = {0};

//     MPI_Get_address(deg_p, &deg_addr);
//     MPI_Get_address(poly_A_p, &poly_A_addr);

//     array_of_displacements[1] = poly_A_addr - deg_addr;

//     MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, input_mpi_t_p);
//     MPI_Type_commit(input_mpi_t_p);

//     return;
// } /* Build_mpi_type_input */

// /*--------------------------------------------------------------------
//  * Function:  Bcast_input
//  * Purpose:   Broadcast the input to all processes using a derived datatype with a single broadcast.
//  */
// void Bcast_input(int source_rank, long long* deg_p, long long* poly_A_p) {
//     MPI_Datatype input_mpi_t;

//     Build_mpi_type_input(deg_p, poly_A_p, &input_mpi_t);
//     MPI_Bcast(deg_p, 1, input_mpi_t, source_rank, MPI_COMM_WORLD); /* process 0 broadcasts */

//     MPI_Type_free(&input_mpi_t);

//     return;
// } /* Bcast_input */