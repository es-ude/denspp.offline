#ifndef FILTER_IIR_TEMPLATE_H
#define FILTER_IIR_TEMPLATE_H
#include <stdint.h>


typedef struct {
    uint8_t coefficient_length;
    double *coefficient;
    uint16_t tap_start;
    uint16_t tap_length;
    void *taps;
} IirFilter;


#define DEF_CALC_IIR(input_type) input_type calc_filter_iir (input_type data, IirFilter *filter) { \
    uint16_t filter_tap_start = filter->tap_start; \
    double* filter_coeff = filter->coefficient; \
    uint8_t filter_coeff_length = filter->coefficient_length; \
    uint8_t filter_tap_length = filter->tap_length; \
    double *filter_tap = filter->taps; \
\
    double val_inp = 0;\
    int16_t pos_tap = filter_tap_length - 1 - filter_tap_start; \
    for(int16_t pos_coeff=0; pos_coeff < filter_coeff_length; pos_coeff++){ \
        if(pos_coeff == 0){ \
            val_inp = data; \
        } else { \
            val_inp -= filter_coeff[pos_coeff] * filter_tap[pos_tap]; \
            pos_tap--; \
            if(pos_tap < 0) pos_tap = filter_tap_length-1; \
        }; \
    } \
    pos_tap = filter_tap_length - 1 - filter_tap_start; \
	double val_out = 0; \
	for (int8_t pos_coeff=filter_coeff_length; pos_coeff < 2 * filter_coeff_length; pos_coeff++){ \
		if (pos_coeff == filter_coeff_length){ \
			val_out = filter_coeff[pos_coeff] * val_inp; \
		} else { \
			val_out += filter_coeff[pos_coeff] * filter_tap[pos_tap]; \
			pos_tap--; \
			if (pos_tap < 0) pos_tap = filter_tap_length-1; \
		}; \
	}; \
\
    filter_tap[filter_tap_start] = val_inp; \
    filter_tap_start++; \
    if(filter_tap_start >= filter_tap_length) filter_tap_start = 0;  \
    filter->tap_start = filter_tap_start; \
    return (input_type)val_out; \
}


#define DEF_NEW_IIR_FILTER_IMPL(id, input_type, coeff_lgth, tap_lgth, ...) \
    static DEF_CALC_IIR(input_type) \
    \
    input_type calc_filter_iir_ ## id (input_type data) { \
        static double filter_taps [tap_lgth] = {0.0}; \
        static double filter_coefficients [] = {__VA_ARGS__}; \
        static IirFilter settings = { \
            .coefficient_length = coeff_lgth, \
            .coefficient = filter_coefficients, \
            .tap_start = 0, \
            .taps = filter_taps, \
            .tap_length = tap_lgth \
        }; \
        return calc_filter_iir(data, & (settings)); \
    }


#define DEF_NEW_IIR_FILTER_PROTO(id, input_type) \
    input_type calc_filter_iir_ ## id (input_type data);


#endif