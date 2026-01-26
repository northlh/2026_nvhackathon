
import numpy as np

def compute_qc_fail_mask(qc_ds, bits):
    mask = sum(1 << (b - 1) for b in bits if b != 1)
    out = None
    for var in qc_ds.data_vars:
        vals = np.nan_to_num(qc_ds[var].values).astype("int32")
        fail = (vals & mask) != 0
        out = fail if out is None else (out | fail)
    return out
