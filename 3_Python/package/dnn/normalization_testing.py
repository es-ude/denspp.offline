import numpy as np
from package.dnn.data_preprocessing import DataNormalization

array = np.array([[0, 8, 2, -20, 18], [9, 3, 6, -2, 5], [0, 5, 10, 15, 20]])

class_data = DataNormalization("FPGA", "", True, False)
result = class_data.normalize(array)
print(array, "\n", result)

"""
def normalize_fpga(frames_in: np.ndarray, do_bipolar=True, do_global=False) -> np.ndarray:
    mean_val = 0 if do_bipolar else 0.5
    scale_mean = 1 if do_bipolar else 2
    scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if do_global else 1

    frames_out = np.zeros(shape=frames_in.shape)

    for i, frame in enumerate(frames_in):
        scale_local = np.max([np.max(frame), -np.min(frame)]) if not do_global else 1
        scale = scale_mean * scale_local * scale_global
        division_value = 1

        while scale > (2**division_value):
            division_value += 1
        print("Scale: ", scale, "Division value: ", 2**division_value)

        maximum = scale_global if do_global else scale_local
        print("Maximum: ", maximum)
        adjust_maximum = maximum
        divider = 2**division_value if do_bipolar else 2**(division_value-1)
        coefficients = [0, 0, 0, 0]
        for j in range(1, 5):
            if adjust_maximum + adjust_maximum/(2**j) <= divider:
                adjust_maximum = adjust_maximum + adjust_maximum/(2**j)
                coefficients[j-1] = 1
        print(coefficients)
        frames_out[i, :] = mean_val + (frame + coefficients[0]*frame/2**1 + coefficients[1]*frame/2**2 + coefficients[2]*frame/2**3 + coefficients[3]*frame/2**4) / (2**division_value)
    return frames_out


result = normalize_fpga(array, False, True)
print(result)
"""
