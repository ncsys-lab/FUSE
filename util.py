import numpy as np


def print_run_stats(esp, esp_10, cts, qual, cyc):
    cts = np.array(cts)
    qual = np.array(qual)
    cyc = np.array(cyc)
    filter_cts = cts[np.where(cts != -1)]

    cts_q1, cts_med, cts_q3 = np.quantile(filter_cts, [0.25, 0.5, 0.75])
    qual_q1, qual_med, qual_q3 = np.quantile(qual, [0.25, 0.5, 0.75])
    cyc_q1, cyc_med, cyc_q3 = np.quantile(cyc, [0.25, 0.5, 0.75])

    print("=========== SUMMARY ===========")
    print("==== Success Probabilities ====")
    print(f"ESP100: {esp:0.3f}")
    print(f"ESP10: {esp_10:0.3f}")

    print("========== CtS Stats ==========")
    print(f"Q1: {cts_q1:.0f}")
    print(f"Med: {cts_med:.0f}")
    print(f"Q3: {cts_q3:.0f}")

    print("===== Sol. Quality Stats ======")
    print(f"Q1: {qual_q1:0.2f}")
    print(f"Med: {qual_med:0.2f}")
    print(f"Q3: {qual_q3:0.2f}")

    print("====== Best Cycle Stats =======")
    print(f"Q1: {cyc_q1:.0f}")
    print(f"Med: {cyc_med:.0f}")
    print(f"Q3: {cyc_q3:.0f}")
