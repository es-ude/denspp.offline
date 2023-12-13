import numpy as np
from package.dnn import data_preprocessing

array = np.array([[0, 8, 2], [9, 3, 6]])

class_data = data_preprocessing.DataNormalization(array, "FPGA", None, False, False)
result = class_data.normalizing()
print(array, "\n", result)

