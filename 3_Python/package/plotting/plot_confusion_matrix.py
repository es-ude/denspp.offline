
from package.plotting.plot_pipeline import results_confusion

#---Start to plot the confusion matrix for two random arrays

cell_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 5]
cell_pred = [0, 1, 0, 0, 1, 1, 0, 1, 2, 4]

results_confusion(cell_true, cell_pred, 0)