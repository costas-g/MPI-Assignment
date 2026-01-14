#ifndef util_matvec_h
#define util_matvec_h

#include <stdio.h>

static long long vectors_diffs(const int* vec_a, const int* vec_b, long long size) {
    if (!vec_a || !vec_b) return -1; /* return an error value if one of the inputs is NULL */

    long long num_errors = 0;
    for (long long i = 0; i < size; i++) {
        if (vec_a[i] != vec_b[i]) {
            num_errors++;
        }
    }
    return num_errors;
}

static void print_matrix(const int *mtx, long long rows, long long cols) {
    if (!mtx) {
        printf("[ERROR]: matrix input is NULL\n");
        return;
    }

    long long i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%d, ", mtx[i*cols+j]);
        }
        puts("");
    }
}

static void print_vector(const int *vec, long long size) {
    if (!vec) {
        printf("[ERROR]: vector input is NULL\n");
        return;
    }

    long long i;
    for (i = 0; i < size; i++) {
        printf("%d, ", vec[i]);
    }
    puts("");
}

#endif