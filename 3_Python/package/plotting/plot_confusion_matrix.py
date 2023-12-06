
from package.plotting.plot_metric import plot_confusion_timestamps, plot_confusion_classes

# Test array
cell_true = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 100],
             [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]]
cell_pred = [[0, 10, 20, 35, 40, 60, 72, 80, 90, 102, 115],
             [0, 1, 2, 3, 1, 2, 2, 3, 0, 1, 0]]


plot_confusion_timestamps(cell_true, cell_pred, False,  r"C:\Users\Leoni Kaiser\Documents\Studium\Master\2. Semester\CPS Projekt\Git_Projekt\spaike_denssp\3_Python\runs")

plot_confusion_classes(cell_true, cell_pred, "pipeline", None, r"C:\Users\Leoni Kaiser\Documents\Studium\Master\2. Semester\CPS Projekt\Git_Projekt\spaike_denssp\3_Python\runs")