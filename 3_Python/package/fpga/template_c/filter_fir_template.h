#ifndef FILTER_FIR_TEMPLATE_H
#define FILTER_FIR_TEMPLATE_H
#include <stdint.h>


typedef struct {
    uint16_t coefficient_length;
    double *coefficients;
    uint16_t tap_start;
    uint16_t tap_length;
    void *taps;
} FirFilter;



#define DEF_CALC_FIR(input_type) input_type calc_filter_fir (input_type data, FirFilter *filter) { \
    uint16_t filter_fir_tap_start = filter->tap_start; \
    double* filter_fir_coeff = filter->coefficients; \
    uint16_t filter_fir_coeff_length = filter->coefficient_length; \
    uint16_t filter_fir_tap_length = filter->tap_length; \
\
    input_type *filter_fir_tap = (input_type *) filter->taps; \
    filter_fir_tap[filter_fir_tap_start] = data; \
\
    double value_mac = 0;\
    int32_t pos_tap = filter_fir_tap_start;\
    for(int16_t pos_coeff=0; pos_coeff < filter_fir_coeff_length; pos_coeff++){ \
        value_mac += filter_fir_coeff[pos_coeff] * filter_fir_tap[pos_tap];\
        pos_tap--; \
        if(pos_tap < 0) pos_tap = filter_fir_tap_length-1; \
    } \
    for(int16_t pos_coeff=filter_fir_tap_length - filter_fir_coeff_length-1; pos_coeff >= 0; pos_coeff--){ \
        value_mac += filter_fir_coeff[pos_coeff] * filter_fir_tap[pos_tap];\
        pos_tap--; \
        if(pos_tap < 0) pos_tap = filter_fir_tap_length-1; \
    } \
\
    filter_fir_tap_start++; \
    if(filter_fir_tap_start >= filter_fir_tap_length) filter_fir_tap_start = 0;  \
    filter->tap_start = filter_fir_tap_start; \
    return (input_type)value_mac; \
}


#define DEF_NEW_FIR_FILTER_IMPL(id, input_type, order, ...) \
    static DEF_CALC_FIR(input_type) \
    \
    input_type calc_filter_fir_ ## id (input_type data) { \
        static input_type fir_filter_taps [order] = {0}; \
        static double fir_filter_coefficients [] = {__VA_ARGS__}; \
        static FirFilter fir_filter = { \
            .coefficient_length = sizeof(fir_filter_coefficients)/sizeof(double), \
            .coefficients = fir_filter_coefficients, \
            .tap_start = 0, \
            .tap_length = order, \
            .taps = fir_filter_taps \
        }; \
        return calc_filter_fir(data, & (fir_filter)); \
    }


#define DEF_NEW_FIR_FILTER_PROTO(id, input_type) \
    input_type calc_filter_fir_ ## id (input_type data);


#endif