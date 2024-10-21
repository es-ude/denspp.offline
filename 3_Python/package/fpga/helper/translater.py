import numpy as np


def get_embedded_datatype(bits_used: int, out_signed=True) -> str:
    """Determine the C datatype for processing data
    Args:
        bits_used:      Integer with bit value
        out_signed:     Boolean if datatype is signed or unsigned
    Return:
        String with datatype in C
    """
    # Define datatyp used in embedded device
    bit_compare = np.array((8, 16, 32, 64))
    used_bitval = np.argwhere((bit_compare / bits_used) - 1 >= 0).flatten()[0]
    return ('' if out_signed else 'u') + 'int' + f'{bit_compare[used_bitval]}' + f'_t'


def slicing_data_intro_string_array(parameter: np.ndarray, slice_value: int) -> str:
    """Slicing data from numpy array into string array
    Args:
        parameter:      Numpy array with parameters
        slice_value:    Slicing value for string array
    Return:
        String array with parameters
    """
    num_ite = int(np.ceil(parameter.size / slice_value))
    params_sliced = ''
    for idx0 in range(num_ite):
        data_row = parameter[idx0 * slice_value:(idx0 + 1) * slice_value] \
            if (idx0 + 1) * slice_value < parameter.size \
            else parameter[idx0 * slice_value:]

        for idx1, val0 in enumerate(data_row):
            params_sliced += f'\t{val0}' + (',' if not (idx0 * 4) + idx1 == parameter.size - 1 else '')
        params_sliced += '\n'
    return params_sliced


def replace_variables_with_parameters(string_input: list, parameters: dict) -> list:
    """Function for search parameter in string list and replace with new defined real values
    Args:
        string_input:   List with input strings from file
        parameters:     Dictionary with parameters (key and value)
    Returns:
        List with corrected string output
    """
    string_output = list()
    for line in string_input:
        if '{$' not in line:
            used_line = line
        else:
            overview_split = line.split('{$')
            used_line = line
            for split_param in overview_split[1:]:
                param_search = split_param.split('}')[0]
                for key, value in parameters.items():
                    if param_search == key:
                        used_line = used_line.replace(f'{{${param_search}}}', value)
                        break
        string_output.append(used_line)
    return string_output


def chck_key_in_list_available(key_chck: str, library: list) -> list:
    """Function for checking if key is in library available
    Args:
        key_chck:   String with key for checking and adding to library
        library:    Library with keys
    Return:
        List of library
    """
    # --- Checking if key in list is available
    value_avai = False
    for key in library:
        if key_chck in key:
            value_avai = True
            break
    # --- Generate output
    if not value_avai:
        library.append(key_chck)
    return library


def generate_params_list(template_list: list, old_library_list=None) -> list:
    """
    Function for generating a parameter list from template files
    Args:
        template_list:      List with template for extracting parameters
        old_library_list:   List with old parameters [Default: None -> Generate new list]
    Returns:
        List with common parameters
    """
    new_params_list = old_library_list if old_library_list is not None else list()

    for line in template_list:
        if '{$' in line:
            overview_split = line.split('{$')
            for split_param in overview_split[1:]:
                param_search = split_param.split('}')[0]
                new_params_list = chck_key_in_list_available(param_search, new_params_list)

    return new_params_list


def read_template_design_file(path2template: str) -> dict:
    """Function for reading a design template file with changed inputs
    Args:
        path2template:  Path to design template file
    Return:
        Dictionary with infos for prototype ['head'] (opt.), implementation ['func'] and used parameters ['params']
    """
    func_temp = list()
    with open(path2template, 'r') as v_handler:
        for line in v_handler:
            func_temp.append(line)
    v_handler.close()
    return {'func': func_temp, 'params': generate_params_list(func_temp)}
