from os.path import join, basename
from glob import glob


def read_mcs_rawdata(file_name: str) -> dict:
    """Function for loading and handling the MCS rawdata from *.mcd file
    :param file_name:   Path to the MCS rawdata file
    :return:            Dictionary with MCS rawdata data
    """
    if '.mcd' not in basename(file_name):
        raise FileNotFoundError(f'File {file_name} not suitable for loading regarding ending.')
    else:
        f = open(file_name, 'rb')
        str_out = [f.readline().decode(encoding="ansi") for idx in range(100)]

        return {'fs': 0.0, 'channel_id': 0.0, 'data': 0.0}


if __name__ == '__main__':
    path_to_data = glob(join("C:/Users/Andre/sciebo/Collab_NeuroRecording/20231116_phase/A1R1a", "*.mcd"))
    path_file = path_to_data[0]
    print(path_file)

    mcs_data = read_mcs_rawdata(path_file)
