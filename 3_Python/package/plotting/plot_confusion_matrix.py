
from package.plotting.plot_metric import prep_confusion

# Test array
cell_true = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 100],
             [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]]
cell_pred = [[0, 10, 20, 35, 40, 60, 72, 80, 90, 102, 115],
             [0, 1, 2, 3, 1, 2, 2, 3, 0, 1, 0]]


prep_confusion(cell_true, cell_pred, "pipeline", "both", True, None, r"C:\Users\Leoni Kaiser\Documents\Studium\Master\2. Semester\CPS Projekt\Git_Projekt\spaike_denssp\3_Python\runs")