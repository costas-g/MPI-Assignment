#ifndef _poly_mult_parallel_h_
#define _poly_mult_parallel_h_

#include <stddef.h> /* defines size_t */

/* Multiplies two polynomials in parallel and returns the result.*/
long long *poly_mult_parallel(const long long *A, size_t deg_A, const long long *B, size_t deg_B, size_t num_threads, double *time);

#endif