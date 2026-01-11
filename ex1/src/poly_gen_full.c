#include <stdio.h>
#include <stdlib.h>

long long *poly_gen_full(size_t degree, int max_coeff){
    long long *poly = malloc((size_t)(degree+1) * sizeof(long long));
    if (!poly) {
        perror("malloc poly");
        exit(EXIT_FAILURE);
    }

    if (max_coeff < 1 || max_coeff > RAND_MAX/2) max_coeff = RAND_MAX;
    else max_coeff = 2*max_coeff;
    for (size_t i = 0; i <= degree; i++){
        int coeff;
        do {
            coeff = rand() % max_coeff - max_coeff/2; /* in range [-max_coeff, max_coeff-1] */
        } while (coeff == 0); /* coefficients should be non-zero integers */
        poly[i] = (long long) coeff;
    }

    return poly;
}