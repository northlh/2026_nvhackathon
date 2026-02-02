
import numpy as np

def evaluate_bins(pred: np.ndarray, obs: np.ndarray, thresholds=[5, 10]):
    """
    Compute per-bin metrics for prediction vs observation arrays.
    Returns dict with 'all' and per-bin stats.
    """
    pred = np.asarray(pred).ravel()
    obs  = np.asarray(obs).ravel()
    diff = pred - obs

    thr = sorted(thresholds)
    edges = [0] + thr + [np.inf]

    results = {}
    results["all"] = {
        "Count": int(len(obs)),
        "MBE": float(np.nanmean(diff)),
        "RMSE": float(np.sqrt(np.nanmean(diff**2))),
    }

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (obs >= lo) & (obs < hi)
        d = diff[m]
        o = obs[m]
        key = f"{lo}-{hi}" if hi != np.inf else f">{lo}"
        if d.size == 0:
            results[key] = {"Count": 0, "MBE": np.nan, "RMSE": np.nan}
        else:
            results[key] = {
                "Count": int(d.size),
                "MBE": float(d.mean()),
                "RMSE": float(np.sqrt(np.mean(d**2))),
            }
    return results


def format_results_table(results: dict) -> str:
    """
    Nicely format the dict produced by evaluate_bins().
    """
    lines = []
    header = f"{'Bin':<12}{'Count':>10}{'MBE':>12}{'RMSE':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for k, v in results.items():
        if k == "all":
            continue
        lines.append(f"{k:<12}{v['Count']:>10}{v['MBE']:>12.5f}{v['RMSE']:>12.5f}")
    lines.append("-" * len(header))
    a = results["all"]
    lines.append(f"{'All':<12}{a['Count']:>10}{a['MBE']:>12.5f}{a['RMSE']:>12.5f}")
    return "\n".join(lines)