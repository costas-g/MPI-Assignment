#ifndef _poly_gen_full_h_
#define _poly_gen_full_h_

#include <stddef.h> /* defines size_t */

/* Generate a full polynomial of with random integer coefficients.*/
long long *poly_gen_full(size_t degree, int max_coeff);

#endif
