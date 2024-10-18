#ifndef WAVEFORM_LUT_TEMPLATE_H
#define WAVEFORM_LUT_TEMPLATE_H
#include <stdint.h>
#include <stdbool.h>


typedef struct {
    uint8_t state;
    uint16_t lut_position;
    uint16_t lut_length;
    void *lut_data;
} WaveformSettings;


#define DEF_GET_LUT_VALUE_FULL(input_type) \
input_type read_waveform_value_runtime (WaveformSettings *filter, bool skip_last_point) { \
    double* lut_values = filter->lut_data; \
    \
    input_type data = lut_values[filter->lut_position]; \
    if((filter->lut_position == filter->lut_length -1) && !skip_last_point){ \
        filter->lut_position = 0; \
        filter->state = 0; \
    } else if((filter->lut_position == filter->lut_length -2) && skip_last_point){ \
        filter->lut_position = 0; \
        filter->state = 0; \
    } else { \
        filter->lut_position++; \
        filter->state = 1; \
    }; \
    return data; \
}


#define DEF_NEW_WAVEFORM_LUT_FULL_IMPL(id, input_type, cnt_type, lut_lgth, ...) \
    static DEF_GET_LUT_VALUE_FULL(input_type) \
    input_type read_waveform_value_runtime_ ## id (bool skip_last_point) { \
        static double lut_data_read [] = {__VA_ARGS__}; \
        static WaveformSettings settings = { \
            .state = 0, \
            .lut_position = 0, \
            .lut_length = lut_lgth, \
            .lut_data = lut_data_read \
        }; \
        return read_waveform_value_runtime(& (settings), skip_last_point); \
    }


#define DEF_GET_LUT_VALUE_OPT(input_type) \
input_type read_waveform_value_runtime (WaveformSettings *filter, bool skip_last_point) { \
    double* lut_values = filter->lut_data; \
    input_type data = 0; \
    \
    if(filter->state == 0){ \
        data = lut_values[filter->lut_position]; \
        filter->lut_position++; \
        if(filter->lut_position == (filter->lut_length - 1)){ \
            filter->state = 1; \
        } else { \
            filter->state = 0; \
        }; \
    } else if(filter->state == 1){ \
        data = lut_values[filter->lut_position]; \
        filter->lut_position--; \
        if(filter->lut_position == 0){ \
            filter->state = 2; \
        } else { \
            filter->state = 1; \
        }; \
    } else if(filter->state == 2){ \
        data = -lut_values[filter->lut_position]; \
        filter->lut_position++; \
        if(filter->lut_position == (filter->lut_length - 1)){ \
            filter->state = 3; \
        } else { \
            filter->state = 2; \
        }; \
    } else if(filter->state == 3){ \
        data = -lut_values[filter->lut_position]; \
        filter->lut_position--; \
        if(filter->lut_position == 0){ \
            if(skip_last_point){ \
                filter->state = 0; \
            } else { \
                filter->state = 4; \
            } \
        } else { \
            filter->state = 3; \
        }; \
    } else if(filter->state == 4){ \
        data = lut_values[0]; \
        filter->lut_position = 0; \
        filter->state = 0; \
    } else { \
        filter->state = 0; \
        filter->lut_position = 0; \
    }; \
    return data; \
}


#define DEF_NEW_WAVEFORM_LUT_OPT_IMPL(id, input_type, cnt_type, lut_lgth, ...) \
    static DEF_GET_LUT_VALUE_OPT(input_type) \
    input_type read_waveform_value_runtime_ ## id (bool skip_last_point) { \
        static double lut_data_read [] = {__VA_ARGS__}; \
        static WaveformSettings settings = { \
            .state = 0, \
            .lut_position = 0, \
            .lut_length = lut_lgth, \
            .lut_data = lut_data_read \
        }; \
        return read_waveform_value_runtime(& (settings), skip_last_point); \
    }


#define DEF_NEW_WAVEFORM_LUT_PROTO(id, input_type, cnt_type) \
    input_type read_waveform_value_runtime_ ## id (bool skip_last_point);


#endif