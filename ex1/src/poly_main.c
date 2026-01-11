#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> /* For strlen             */
#include <mpi.h>    /* For MPI functions, etc */ 

#include "poly_gen_full.h"
#include "poly_mult_serial.h"
#include "poly_mult_parallel.h"
#include "poly_util.h"

const int MAX_STRING = 1024;

void Usage(char* prog_name);

int main(int argc, char* argv[]) {
    int comm_sz;                    /* Number of processes  */
    int my_rank;                    /* My process rank      */
    long long degree;               /* Polynomials' degree  */
    long long deg_A, deg_B, deg_R;  /* polynomial degree    */

    /* Parse inputs and error check */
    if (argc < 2) Usage(argv[0]);
    degree  = strtoll(argv[1], NULL, 10); if (degree < 1) Usage(argv[0]);
    deg_A = deg_B = degree;
    deg_R = 2*degree;

    /* Timing variables */
    struct timespec start, finish;
    double elapsed_time, gen_time;
    srand((unsigned int) time(NULL)); /* seed random generator */
    // struct xorshift32_state prng_state;
    // prng_state.a = (unsigned int) time(NULL); /* seed the PRNG */


    /* ---------------------------------------- Main body ---------------------------------------- */
    printf("Multiplication of two %lld-degree polynomials using %d processes.\n", degree, comm_sz);

    /* -------------------- Generate the two polynomials ---------------------- */
    printf("\n================================================");
    printf("\nGenerating Polynomials...\n");
    int max_coeff = 9;          /* maximum coefficient value (absolute value) */
    long long *poly_A, *poly_B; /* constant input polynomials A, B*/
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        poly_A = poly_gen_full((size_t) deg_A, max_coeff);
        poly_B = poly_gen_full((size_t) deg_B, max_coeff);
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */

    /* elapsed time */
    gen_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Polynomials generation time (s): %9.6f\n", gen_time);

    
    /* ---------------------- Serial Poly Multiplication ---------------------- */
    printf("\n================================================");
    printf("\nSerial Poly Multiplication...\n");
    long long *poly_R_serial = NULL;
    poly_R_serial = calloc((size_t) deg_R + 1, sizeof(long long));
    if (!poly_R_serial) {
        perror("calloc poly_R_serial");
        exit(EXIT_FAILURE);
    }
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        poly_R_serial = poly_mult_serial(poly_A, deg_A, poly_B, deg_B, &elapsed_time);
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Serial poly mult time (s):   %9.6f\n", elapsed_time);
    double serial_time = elapsed_time;

    #ifdef DEBUG
    print_poly(poly_A, deg_A);
    print_poly(poly_B, deg_B);
    print_poly(poly_R_serial, deg_R);
    #endif


    /* ---------------------- Parallel Poly Multiplication ---------------------- */
    printf("\n================================================");
    printf("\nParallel Poly Multiplication...\n");

    /* No MPI calls before this */
    /* Start up MPI */
    MPI_Init(&argc, &argv); 

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    
    char hostname[512];
    gethostname(hostname, 512);

    if (my_rank != 0) { 
        /* Create message */
        // sprintf(greeting, "Greetings from process %d of %d - %s!", my_rank, comm_sz, hostname);
        /* Send message to process 0 */
        // MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD); 
    } else {
        // MPI_Status status;
        // int count;
        /* Print my message */
        // printf("Greetings from process %d of %d - %s!\n", my_rank, comm_sz, hostname);
        for (int q = 1; q < comm_sz; q++) {
            /* Receive message from process q */
            // MPI_Recv(greeting, MAX_STRING, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // MPI_Get_count(&status, MPI_CHAR, &count);
            /* Print message from process q */
            // printf("[Received from %d, tag=%d, count=%d] %s\n", status.MPI_SOURCE, status.MPI_TAG, count, greeting);
        } 
    }

    long long *poly_R_parallel = NULL;
    poly_R_parallel = calloc((size_t) 2*degree + 1, sizeof(long long));
    if (!poly_R_parallel) {
        perror("calloc poly_R_parallel");
        exit(EXIT_FAILURE);
    }
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
        // poly_R_parallel = poly_mult_parallel(poly_A, deg_A, poly_B, deg_B, comm_sz, &elapsed_time);
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */
    elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9; 
    printf("  Parallel poly mult time (s): %9.6f\n", elapsed_time);
    double parallel_time = elapsed_time;


    /* --------- Speedup calculation --------- */ 
    printf("\nSpeedup: %.3f\n", serial_time/parallel_time);
    printf("\n");


    /* ---------------------- Confirm correctness ---------------------- */
    printf("\n================================================");
    printf("\nComparing Serial & Parallel poly mult results...\n");
    long long nerrors;
    nerrors = poly_count_errors(poly_R_parallel, poly_R_serial, degree);
    if (nerrors == 0) {
        printf("  Results match!\n");
    } else {
        printf("  ERROR: Results mismatch! # of errors = %lld\n", nerrors);
    }


    /* ------------------------------------ Cleanup ------------------------------------ */
    /* Free allocated memory */
    free(poly_R_serial);
    free(poly_R_parallel);
    free(poly_A);
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
void Usage(char *prog_name) {
   fprintf(stderr, "Usage: %s <degree>\n", prog_name);
   fprintf(stderr, "   degree: Degree of the polynomials. Must be positive.\n");
   //fprintf(stderr, "   thread_count: Number of threads. Should be positive.\n");
   exit(0);
}  /* Usage */