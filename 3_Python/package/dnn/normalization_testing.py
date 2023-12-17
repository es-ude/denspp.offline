import numpy as np
from package.dnn.data_preprocessing import DataNormalization

array = np.array([[0, 8, 2], [9, 3, 6]])

class_data = DataNormalization(array, "CPU", "minmax", True, False)
result = class_data.normalize()
print(array, "\n", result)

