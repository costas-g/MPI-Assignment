#ifndef xorshift32_h_
#define xorshift32_h_

#include <stdint.h>

struct xorshift32_state {
    uint32_t a;
};

/* The state must be initialized to non-zero */
static inline uint32_t xorshift32(struct xorshift32_state *state){
	/* Algorithm "xor" from p. 4 of George Marsaglia, "Xorshift RNGs" */
	uint32_t x = state->a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return state->a = x;
}

#endif