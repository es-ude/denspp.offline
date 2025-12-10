import numpy as np
import os
from glob import glob


def chck_file_temp(path2temp: str) -> None:
    """"""
    # --- Check if folder is available
    if not os.path.exists(path2temp):
        os.makedir(path2temp)
    else:
        list_dir = glob(os.path.join(path2temp, '*.txt'))
        for name in list_dir:
            os.remove(name)    

    
def write_ext_file(data_list: list, ch_list: list, path2temp_data: str, file_name_default='temp_data', do_python=False) -> str:
    """"""         
    str_option = 'w'
    # --- Check if folder is available
    if not os.path.exists(path2temp_data):
        os.makedirs(path2temp_data)

    # --- Copy data into file
    for ch_idx, ch_num in enumerate(ch_list):
        data = data_list[ch_num] if do_python else data_list[ch_idx]
                
        write_file_name = f'{file_name_default}{ch_num:02d}.txt'
        write_file_path = os.path.join(path2temp_data, write_file_name)
        with open(write_file_path, str_option) as file:
            for value in data:
                if isinstance(value, np.bool_):
                    file.write(f"{1}\n" if value else f"{0}\n")
                elif isinstance(value, bytes):
                    file.write(f"{int.from_bytes(value, 'big', signed=False)}\n")
                else:
                    file.write(f"{value}\n")
        
    return write_file_path


def process_data_rotation(data_in: list) -> list:
    """Rotation from timeseries to channel per list element"""
    num_ch = len(data_in)
    num_length = data_in[0].size
    num_type = data_in[0].dtype
    print(f"Array has size of {num_ch} with {num_length} elemens")

    data_out = list()
    data_send = np.zeros(shape=(num_length, num_ch), dtype=num_type)
    for chn, series in enumerate(data_in):
        data_send[:, chn] = series

    for chn, series in enumerate(data_send):
        data_out.append(series)

    return data_out


def translate_data_float2int(data_in: list, dac_bit: int, dac_signed: bool) -> list:
    # max_dac_output define the Bit depth of the DAC 12bit -> 2^12 = 4096; signed -> 2048 
    """"""
    data_out = list()
    
    max_dac_output =(2** dac_bit) / 2 if dac_signed else (2** dac_bit)
    
    for data in data_in:
        # Find the absolute maximum value in the signal (regardless of positive or negative) 
         # With this call The loudest point in the signal (whether positive or negative) corresponds exactly to the target value max_dac_output.
        val_abs_max = data.max() if data.max() > abs(data.min()) else data.min()
        if val_abs_max < 0:
            val_abs_max = abs(val_abs_max)

        scaled_data = max_dac_output / val_abs_max * data
        if dac_signed:
            clipped_data = np.clip(scaled_data, -32768, 32767) # Prevent overflow for signed 16-bit integer
        else:
            clipped_data = np.clip(scaled_data, 0, 65535) # Prevent overflow for unsigned 16-bit integer

        data_out.append(np.array(clipped_data, dtype=np.int16)) # Convert to int16

    if np.any(data_out[0] > 32767) or np.any(data_out[0] < -32768):
        print("WARNUNG: Overflow erkannt!")
        print(f"Max: {np.max(data_out[0])}, Min: {np.min(data_out[0])}")
    else:
        print("Alle Werte im gültigen Bereich")
    return data_out