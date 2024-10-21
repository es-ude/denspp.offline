#ifndef FILTER_FIR_ALL_TEMPLATE_H
#define FILTER_FIR_ALL_TEMPLATE_H
#include <stdint.h>


typedef struct {
    uint16_t tap_start;
    uint16_t tap_length;
    uint16_t pos_end;
    bool first_run;
    void *taps;
} FirAllFilter;



#define DEF_CALC_FIR_ALLPASS(input_type) \
input_type calc_filter_fir_all (input_type data, FirAllFilter *filter) { \
    uint16_t filter_tap_start = filter->tap_start; \
\
    input_type *filter_tap = (input_type *) filter->taps; \
    filter_tap[filter_tap_start] = data; \
\
    filter_tap_start++; \
    if(filter_tap_start >= filter->tap_length){ \
        filter_tap_start = 0; \
        filter->first_run = true; \
    } \
    filter->tap_start = filter_tap_start; \
\
    if(filter->first_run && filter->tap_start != 0){ \
        filter->pos_end = filter->tap_start -1; \
    } else { \
        filter->pos_end = filter->tap_length -1; \
    } \
    return filter_tap[pos_end]; \
}


#define DEF_NEW_FIR_ALL_FILTER_IMPL(id, input_type, order, ...) \
    static DEF_CALC_FIR_ALL(input_type) \
    \
    input_type calc_filter_fir_all_ ## id (input_type data) { \
        static input_type filter_taps [order] = {0}; \
        static FirAllFilter settings = { \
            .pos_end = order-1, \
            .tap_start = 0, \
            .tap_length = order, \
            .taps = filter_taps, \
            .first_run = false \
        }; \
        return calc_filter_fir_all(data, & (settings)); \
    }


#define DEF_NEW_FIR_FILTER_PROTO(id, input_type) \
    input_type calc_filter_fir_all_ ## id (input_type data);


#endif