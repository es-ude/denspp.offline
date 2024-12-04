#ifndef FILTER_FIR_ALL_TEMPLATE_H
#define FILTER_FIR_ALL_TEMPLATE_H
#include <stdint.h>
#include <stdbool.h>


typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    uint16_t pos_end;
    bool first_run_done;
    void *taps;
} FirAllFilter;



#define DEF_CALC_FIR_ALLPASS(input_type) \
input_type calc_filter_fir_all (input_type data, FirAllFilter *filter) { \
    input_type *filter_tap = (input_type *) filter->taps; \
    filter_tap[filter->tap_start] = data; \
\
    if(filter->tap_start >= filter->tap_length -1){ \
        filter->tap_start = 0; \
        filter->first_run_done = true; \
    } else { \
        filter->tap_start++; \
        filter->first_run_done = filter->first_run_done; \
    } \
\
    if(filter->first_run_done){ \
        filter->pos_end = filter->tap_start; \
    } else { \
        filter->pos_end = filter->tap_length -1; \
    } \
    return filter_tap[filter->pos_end]; \
}


#define DEF_NEW_FIR_ALL_FILTER_IMPL(id, input_type, order) \
    static DEF_CALC_FIR_ALLPASS(input_type) \
    \
    input_type calc_filter_fir_all_ ## id (input_type data) { \
        static input_type filter_taps [order] = {0}; \
        static FirAllFilter settings = { \
            .pos_end = order-1, \
            .tap_start = 0, \
            .tap_length = order, \
            .taps = filter_taps, \
            .first_run_done = false \
        }; \
        return calc_filter_fir_all(data, & (settings)); \
    }


#define DEF_NEW_FIR_ALL_FILTER_PROTO(id, input_type) \
    input_type calc_filter_fir_all_ ## id (input_type data);


#endif