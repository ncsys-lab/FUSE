import numpy as np


def print_cts_stats(arr):
    arr_len = len(arr)
    arr = np.array(arr)
    filter_cts = arr[np.where(arr != -1)]
    q1, med, q3 = np.quantile(filter_cts, [0.25, 0.5, 0.75])
    print("==== CtS Statistics ====")
    print(f"Q1: {q1:0.2f}")
    print(f"Median: {med:0.2f}")
    print(f"Q3: {q3:0.2f}")
