#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long  *poly_mult_parallel(const long long *A, size_t n, const long long *B, size_t m, size_t thread_count, double *time){
    struct timespec start, end;
    size_t r = n + m + 1;

    long long *R_global = calloc(r, sizeof(long long)); 
    if (!R_global) {
        perror("calloc R_global");
        exit(EXIT_FAILURE);
    }

    long long **R_locals = malloc(thread_count * sizeof(*R_locals));
    if (!R_locals) {
        perror("malloc R_locals");
        exit(EXIT_FAILURE);
    }

    long long *R_local;
    long long coeff_prod;
    clock_gettime(CLOCK_MONOTONIC, &start); /* start time */
    # pragma omp parallel num_threads(thread_count) private(R_local, coeff_prod)
    {   
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        // Each thread allocates its private array
        R_locals[tid] = calloc(r, sizeof(long long));
        if (!R_locals[tid]) {
            perror("calloc R_local");
            exit(EXIT_FAILURE);
        }
        R_local = R_locals[tid]; /* pointer assignment - alias */

        # pragma omp for nowait /* nowait removes the implicit barrier after the for block*/
        for (size_t i = 0; i <= n; i++){
            for(size_t j=0; j <= m ; j++){
                coeff_prod = A[i] * B[j];
                R_local[i+j] += coeff_prod; /* per-thread private memory */
            }
        } /* implicit barrier */

        /* Combine results */
        size_t k = tid * r/thread_count; /* OPTIONAL: start from evenly spaced out indexes so that contention is reduced */
        for (size_t i = 0; i < r; i++, k++){
            if (k >= r)
                k = 0; /* wrap around to the head of the array */

            /* critical section */
            # pragma omp atomic /* use of atomic for potential performance gains if CPU supports load-modify-store instructions */
            R_global[k] += R_local[k]; /* safely update the shared variable */
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end); /* end time */

    for (size_t t = 0; t < thread_count; t++) {
        // /* Combine results sequentially */
        // for (size_t i = 0; i < r; i++) {
        //     R_global[i] += R_locals[t][i];
        // }

        /* Free allocated memory */
        free(R_locals[t]);
    }

    /* Free allocated memory */
    free(R_locals);
    
    /* Elapsed time */
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    *time = time_spent;

    return R_global;
}