// #define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long  *poly_mult_serial(const long long *A, size_t deg_A, const long long *B, size_t deg_B, double *time) {
    size_t deg_R = deg_A + deg_B; /* resultant degree */
    long long *R = calloc((size_t) deg_R + 1, sizeof(long long));
    if(!R) {
        perror("calloc R");
        exit(EXIT_FAILURE);
    }

    struct timespec start, finish;

    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    for (size_t i = 0; i <= deg_A; i++){
        for(size_t j = 0; j <= deg_B; j++){
            R[i+j] += A[i] * B[j];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish); /* finish time */

    double time_spent = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9;
    *time = time_spent;

    return R;
}