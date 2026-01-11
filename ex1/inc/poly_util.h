#ifndef poly_util_h
#define poly_util_h

#include <stdio.h>

/* Counts the number of errors in matching two polynomials' coefficients.*/
static long long poly_count_errors(const long long* poly_a, const long long* poly_b, long long deg) {
    long long num_errors = 0;
    for (long long i = 0; i < deg+1; i++) {
        if (poly_a[i] != poly_b[i]) {
            num_errors++;
        }
    }
    return num_errors;
}

#ifdef DEBUG
static void print_poly(const long long *poly, long long deg) {
    long long i;
    for (i = 0; i < deg+1; i++) {
        printf("%lld, ", poly[i]);
    }
    puts("");
}
#endif

#endif