#ifndef _poly_mult_parallel_h_
#define _poly_mult_parallel_h_

#include <stddef.h> /* defines size_t */

/* Multiplies one full polynomial with one partial polynomial and writes the partial result to a full-sized output. */
void poly_mult_partial(const long long *A_part, size_t deg, const long long *B_full, size_t local_terms, long long *R_part, size_t my_rank);

#endif