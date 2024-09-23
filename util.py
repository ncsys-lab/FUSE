import numpy as np


def print_run_stats(esp, esp_10, cts, qual):
    cts = np.array(cts)
    qual = np.array(qual)
    filter_cts = cts[np.where(cts != -1)]
    cts_q1, cts_med, cts_q3 = np.quantile(filter_cts, [0.25, 0.5, 0.75])
    qual_q1, qual_med, qual_q3 = np.quantile(qual, [0.25, 0.5, 0.75])
    print(
        f"{esp:0.3f}\t {esp_10:0.3f}\t {cts_q1:0.2f}\t {cts_med:0.2f}\t {cts_q3:0.2f}\t {qual_q1:0.2f}\t {qual_med:0.2f}\t {qual_q3:0.2f}"
    )
    """
    print("==== CtS Statistics ====")
    print(f"Q1: {q1:0.2f}")
    print(f"Median: {med:0.2f}")
    print(f"Q3: {q3:0.2f}")
    """
