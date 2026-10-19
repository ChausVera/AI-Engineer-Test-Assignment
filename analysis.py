"""Check whether the hourly flotation data reproduces the published lag correlations.

Usage:  python analysis.py [path/to/flotation_process_data_hourly.csv]

Reference values (research on the Kaggle "Quality Prediction in a Mining Process" data):
  amina now   -> silica later : 1h 0.145, 6h 0.052
  silica earlier -> amina now : 1h 0.194, 2h 0.230, 3h 0.255
"""
import sys

import numpy as np
import pandas as pd

AMINA = "Amina Flow"
SILICA = "% Silica Concentrate"
LAGS = [1, 2, 3, 4, 6, 12]
TOL = 0.02  # how far from the research value still counts as a match

EXPECTED = {
    "amina -> silica": {1: 0.145, 6: 0.052},
    "silica -> amina": {1: 0.194, 2: 0.230, 3: 0.255},
}


def load(path):
    df = pd.read_csv(path)
    if df[AMINA].dtype == object:  # decimal-comma version of the file
        df = pd.read_csv(path, decimal=",")
    df["date"] = pd.to_datetime(df["date"])
    return df


def quality_report(df):
    print("=== DATA QUALITY ===")
    print(f"rows: {len(df)}   range: {df['date'].min()} -> {df['date'].max()}")
    dup = int(df["date"].duplicated().sum())
    print(f"duplicate timestamps: {dup}")
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="h")
    missing = len(full.difference(df["date"]))
    print(f"missing hours inside the range: {missing} of {len(full)}")
    print(f"NaN in {AMINA}: {int(df[AMINA].isna().sum())}, in {SILICA}: {int(df[SILICA].isna().sum())}")

    gaps = df["date"].drop_duplicates().sort_values().diff()
    big = gaps[gaps > pd.Timedelta(hours=1)]
    for idx, g in big.items():
        print(f"  gap of {g} ending at {df.loc[idx, 'date']}")

    print("columns where one value fills >50% of rows (possible stuck sensor / held lab value):")
    found = False
    for c in df.columns.drop("date"):
        share = df[c].value_counts(normalize=True, dropna=True).iloc[0]
        if share > 0.5:
            print(f"  {c}: {share:.0%}")
            found = True
    if not found:
        print("  none")
    print()


def lag_table(df):
    # full hourly index, so a gap in the data is not treated as "the next hour"
    d = df.drop_duplicates("date").set_index("date").asfreq("h")
    a, s = d[AMINA], d[SILICA]
    rows = []
    for h in LAGS:
        # amina now vs silica h hours LATER
        rows.append(("amina -> silica", h, a, s.shift(-h)))
        # amina now vs silica h hours EARLIER
        rows.append(("silica -> amina", h, a, s.shift(h)))
    out = []
    for direction, h, x, y in rows:
        pair = pd.concat([x, y], axis=1).dropna()
        r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        exp = EXPECTED[direction].get(h)
        diff = None if exp is None else r - exp
        verdict = "" if exp is None else ("MATCH" if abs(diff) <= TOL else "DIFFERS")
        out.append((direction, h, r, rho, len(pair), exp, diff, verdict))
    return out


def granger(y, x, p):
    """Does x's past (p hourly lags) improve prediction of y beyond y's own past?

    Restricted model: y(t) ~ const + y(t-1..t-p)
    Full model:       y(t) ~ const + y(t-1..t-p) + x(t-1..t-p)
    Returns F statistic, p-value (if scipy is installed), gain in R^2, n.
    """
    d = pd.DataFrame({"y": y})
    for k in range(1, p + 1):
        d[f"y{k}"] = y.shift(k)
        d[f"x{k}"] = x.shift(k)
    d = d.dropna()  # rows touching the March gap drop out here
    Y = d["y"].to_numpy()
    ones = np.ones((len(d), 1))
    Xr = np.hstack([ones, d[[f"y{k}" for k in range(1, p + 1)]].to_numpy()])
    Xf = np.hstack([Xr, d[[f"x{k}" for k in range(1, p + 1)]].to_numpy()])

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return float(((Y - X @ beta) ** 2).sum())

    rss_r, rss_f = rss(Xr), rss(Xf)
    tss = float(((Y - Y.mean()) ** 2).sum())
    n, kf = len(d), Xf.shape[1]
    F = ((rss_r - rss_f) / p) / (rss_f / (n - kf))
    try:
        from scipy.stats import f as fdist
        pval = float(fdist.sf(F, p, n - kf))
    except ImportError:
        pval = float("nan")
    return F, pval, (rss_r - rss_f) / tss, n


def granger_table(df):
    d = df.drop_duplicates("date").set_index("date").asfreq("h")
    series = {
        "levels": (d[AMINA], d[SILICA]),
        # differencing removes slow shared drift, which can fake a lead/lag
        "differences": (d[AMINA].diff(), d[SILICA].diff()),
    }
    out = []
    for transform, (a, s) in series.items():
        for p in (3, 6):
            for label, (y, x) in {"silica -> amina": (a, s), "amina -> silica": (s, a)}.items():
                F, pval, gain, n = granger(y, x, p)
                out.append((label, transform, p, F, pval, gain, n))
    return out


def print_granger(df):
    print()
    print("=== GRANGER-STYLE CHECK (does the past of X add to predicting Y beyond Y's own past?) ===")
    print(f"{'direction':16} {'data':11} {'lags':>4} {'F':>8} {'p-value':>9} {'R2 gain':>8} {'n':>6}")
    res = granger_table(df)
    for label, transform, p, F, pval, gain, n in res:
        print(f"{label:16} {transform:11} {p:>4} {F:>8.1f} {pval:>9.2g} {gain:>8.4f} {n:>6}")

    gains = {(lab, tr, p): g for lab, tr, p, _, _, g, _ in res}
    ratios = []
    for tr in ("levels", "differences"):
        for p in (3, 6):
            fwd, rev = gains[("silica -> amina", tr, p)], gains[("amina -> silica", tr, p)]
            ratios.append(fwd / rev if rev > 0 else float("inf"))
    print()
    print("Ratio of R2 gain, silica->amina / amina->silica (levels 3, 6; differences 3, 6): "
          + ", ".join(f"{r:.1f}x" for r in ratios))
    if all(r >= 2 for r in ratios):
        print("Heuristic verdict: silica's past predicts amina clearly better than the reverse, "
              "also after differencing -> consistent with reactive dosing.")
    elif ratios[0] >= 2 and ratios[2] < 2:
        print("Heuristic verdict: the lead appears in levels but not in differences -> "
              "may be shared slow drift; do not claim reactive dosing.")
    else:
        print("Heuristic verdict: inconclusive; do not claim a direction.")
    print("Note: residuals are autocorrelated, so p-values are optimistic; rely on R2 gain and the "
          "levels-vs-differences comparison. Granger 'causality' means predictive lead only.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/flotation_process_data_hourly.csv"
    df = load(path)
    quality_report(df)

    print("=== LAG CORRELATIONS (Pearson r; Spearman as robustness check) ===")
    print(f"{'direction':16} {'lag_h':>5} {'r':>7} {'spearman':>9} {'n':>6} {'research':>9} {'diff':>7}  verdict")
    res = lag_table(df)
    for direction, h, r, rho, n, exp, diff, verdict in res:
        exp_s = "" if exp is None else f"{exp:.3f}"
        diff_s = "" if diff is None else f"{diff:+.3f}"
        print(f"{direction:16} {h:>5} {r:>7.3f} {rho:>9.3f} {n:>6} {exp_s:>9} {diff_s:>7}  {verdict}")

    sa = {h: r for d, h, r, *_ in res if d == "silica -> amina"}
    asil = {h: r for d, h, r, *_ in res if d == "amina -> silica"}
    peak = max(sa, key=lambda k: abs(sa[k]))
    print()
    print(f"silica -> amina peaks at lag {peak}h (r = {sa[peak]:.3f}); research peak is 3h")
    print(f"silica -> amina at 3h vs amina -> silica at 3h: {sa[3]:.3f} vs {asil[3]:.3f}")
    checked = [v for *_, v in res if v]
    print(f"matches: {checked.count('MATCH')} of {len(checked)} reference values (tolerance +/-{TOL})")

    print_granger(df)


if __name__ == "__main__":
    main()