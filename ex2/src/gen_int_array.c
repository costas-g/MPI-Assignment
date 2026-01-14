#include <stdio.h>
#include <stdlib.h>

#include "gen_int_array.h"

int gen_int_array(int *arr_out, long long size, int max_val){
    if (max_val < 1) max_val = RAND_MAX;
    for (long long i = 0; i < size; i++){
        arr_out[i] = rand() % max_val + 1; /* in range [1, max_val] */
    }

    return 0;
}