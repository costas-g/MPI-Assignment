#ifndef _poly_mult_serial_h_
#define _poly_mult_serial_h_

#include <stddef.h> /* defines size_t */

/* Multiplies two polynomials and returns the result.*/
long long *poly_mult_serial(const long long *A, size_t deg_A, const long long *B, size_t deg_B, double *time);

#endif