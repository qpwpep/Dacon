import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins_csv", required=True)
    ap.add_argument("--sat_csv", required=True)
    args = ap.parse_args()

    bins = pd.read_csv(args.bins_csv)
    sat = pd.read_csv(args.sat_csv)

    # parse bin lows
    bins["bin_low"] = bins["dist_bin"].str.extract(r"\[(.*?),").astype(float)
    bins["bin_high"] = bins["dist_bin"].str.extract(r",\s*(.*?)\)").astype(float)

    valid_bins = bins[bins["split"] == "valid"].copy()

    print("\n=== VALID: tail share (by fold) ===")
    # total valid n per fold from bins (sum over bins)
    valid_n = valid_bins.groupby("fold")["n"].sum()

    for thr in [50, 60, 70, 80]:
        tail = valid_bins[valid_bins["bin_low"] >= thr].groupby("fold")[["n","frac"]].sum()
        tail = tail.rename(columns={"n": f"tail>={thr}_n", "frac": f"tail>={thr}_frac"})
        out = pd.concat([valid_n.rename("valid_n"), tail], axis=1).fillna(0)
        out[f"tail>={thr}_frac"] = out[f"tail>={thr}_frac"].astype(float)
        print(f"\n-- dist >= {thr}m --")
        print(out.sort_values(f"tail>={thr}_frac", ascending=False).to_string())

    # bin-wise maxima
    print("\n=== VALID: which fold is MAX for each dist_bin (by frac) ===")
    idx = valid_bins.groupby("dist_bin")["frac"].idxmax()
    max_by_bin = valid_bins.loc[idx, ["dist_bin","fold","frac","n"]].sort_values("dist_bin")
    print(max_by_bin.to_string(index=False))

    # fold1 vs others on the “problem bins”
    pivot_frac = valid_bins.pivot(index="fold", columns="dist_bin", values="frac")
    pivot_n = valid_bins.pivot(index="fold", columns="dist_bin", values="n")

    focus_bins = ["[70.0, 80.0)", "[80.0, 1000000000.0)"]
    for b in focus_bins:
        if b in pivot_frac.columns:
            f1 = pivot_frac.loc[1, b]
            others = pivot_frac.drop(index=1)[b].mean()
            ratio = (f1 / others) if others > 0 else np.inf
            print(f"\n=== Focus bin {b} (VALID) ===")
            print(f"fold1 frac={f1:.6f}, others_mean frac={others:.6f}, ratio={ratio:.2f}")
            print(f"fold1 n={int(pivot_n.loc[1,b])}, others mean n={pivot_n.drop(index=1)[b].mean():.2f}")

    # sat95
    print("\n=== SAT95 (VALID) ===")
    sat_valid = sat[sat["split"] == "valid"].sort_values("fold")
    print(sat_valid[["fold","n","sat95_n","sat95_rate","sat95_dist_p50","sat95_seq_len_mean"]].to_string(index=False))

if __name__ == "__main__":
    main()
