#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "poly_mult_parallel.h"

void poly_mult_partial(const long long *A_part, size_t deg, const long long *B_full, size_t local_terms, long long *R_part, size_t my_rank) {
    /* B_full is full-sized. A_part has only local_terms size. */
    /* R_part is full-sized but only the partial result is computed. */
    size_t offset; /* it's the minimum degree for which a coefficient is computed by each process */

    /* Each process needs to correctly write the partial results at the corresponding positions. 
     * It works because each rank gets its corresponding block of B in rank order. 
     */
    offset = local_terms * my_rank;

    for (size_t i = 0; i < local_terms; i++){
        for(size_t j = 0; j <= deg ; j++){
            R_part[i + j + offset] += A_part[i] * B_full[j];
        }
    }

    return;
}