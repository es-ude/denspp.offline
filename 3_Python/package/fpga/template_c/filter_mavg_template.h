#ifndef FILTER_MAVG_TEMPLATE_H
#define FILTER_MAVG_TEMPLATE_H
#include <stdint.h>


typedef struct {
    double coefficients;
    uint16_t tap_start;
    uint16_t tap_length;
    void *taps;
} MavgFilter;



#define DEF_CALC_MAVG(input_type) \
input_type calc_moving_average (input_type data, MavgFilter *filter) { \
    uint16_t filter_tap_start = filter->tap_start; \
    uint16_t filter_tap_length = filter->tap_length; \
\
    input_type *filter_tap = (input_type *) filter->taps; \
    filter_tap[filter_tap_start] = data; \
\
    double value_mac = 0;\
    int16_t pos_tap = filter_tap_start;\
    for(uint16_t pos_coeff=0; pos_coeff < filter_tap_length; pos_coeff++){ \
        value_mac += filter->coefficients * filter_tap[pos_tap];\
        pos_tap--; \
        if(pos_tap < 0) pos_tap = filter_tap_length-1; \
    } \
\
    filter_tap_start++; \
    if(filter_tap_start >= filter_tap_length) filter_tap_start = 0;  \
    filter->tap_start = filter_tap_start; \
    return (input_type)value_mac; \
}


#define DEF_NEW_MAVG_FILTER_IMPL(id, input_type, order, coeff) \
    static DEF_CALC_MAVG(input_type) \
    \
    input_type calc_moving_average_ ## id (input_type data) { \
        static input_type filter_taps [order] = {0}; \
        static MavgFilter filter_params = { \
            .coefficients = coeff, \
            .tap_start = 0, \
            .tap_length = order, \
            .taps = filter_taps \
        }; \
        return calc_moving_average (data, & (filter_params)); \
    }


#define DEF_NEW_MAVG_FILTER_PROTO(id, input_type) \
    input_type calc_moving_average_ ## id (input_type data);


#endif