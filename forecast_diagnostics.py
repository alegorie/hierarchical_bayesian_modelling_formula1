"""Розклад різниці прогнозу primary_train − pooled_train (fixed origin).

Читає forecast_laps.csv і forecast_stints.csv, які пише f1deg.forecast.evaluate_models.
Усі різниці — A − B, де A = primary_train, B = pooled_train: > 0 означає, що primary гірша.
bias = медіана прогнозу − спостереження (> 0: прогноз повільніший за реальне коло).

Запуск із командного рядка:
    python forecast_diagnostics.py "D:\\rerun\\f1_tyre_degradation_revision\\outputs_local\\full\\forecast"

У ноутбуці (додатково рахує, скільки гонок / стінтів було в train для траси й комірки траса×компаунд):
    from forecast_diagnostics import run
    res = run(cfg.out_dir / "forecast", all_stints=pl.dataset().stints)

Нічого в outputs пайплайна не змінює; таблиці пише в ./forecast_diagnostics/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

MODEL_A, MODEL_B = "primary_train", "pooled_train"
SCHEME = "fixed_origin"
OLS = "ols_fuel_ext"


# --------------------------------------------------------------------------- helpers
def _year_of(race: pd.Series) -> pd.Series:
    return pd.to_numeric(race.astype(str).str.extract(r"((?:19|20)\d{2})", expand=False), errors="coerce")


def _cluster_mean(d: pd.Series, cluster: pd.Series):
    """Середнє по колах і cluster-robust SE з гонкою як кластером."""
    v = d.to_numpy(float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    m = v.mean()
    s = pd.Series(v - m).groupby(cluster.to_numpy()).sum().to_numpy()
    G = len(s)
    se = float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n) if G > 1 else np.nan
    return float(m), se, G


def _show(title: str, df: pd.DataFrame, note: str = "") -> None:
    print(f"\n=== {title}")
    if note:
        print(note)
    if df is None or len(df) == 0:
        print("(немає даних)")
        return
    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.max_rows", 200):
        print(df.round(4).to_string())


# --------------------------------------------------------------------------- core tables
def load(forecast_dir) -> tuple:
    fd = Path(forecast_dir)
    laps = pd.read_csv(fd / "forecast_laps.csv")
    stints = pd.read_csv(fd / "forecast_stints.csv")
    summary = pd.read_csv(fd / "forecast_summary.csv") if (fd / "forecast_summary.csv").exists() else None
    return laps, stints, summary


def paired_laps(laps: pd.DataFrame, stints: pd.DataFrame, a: str = MODEL_A, b: str = MODEL_B, scheme: str = SCHEME) -> pd.DataFrame:
    fx = laps[laps["scheme"] == scheme].copy()
    fx["err"] = fx["pred_median"] - fx["y"]
    key = ["stint_code", "lap_in_stint"]
    A = fx[fx["model"] == a].set_index(key)
    B = fx[fx["model"] == b].set_index(key)
    if A.empty or B.empty:
        raise ValueError(f"немає рядків для {a!r} або {b!r}; моделі у файлі: {sorted(fx['model'].unique())}")
    if A.index.has_duplicates or B.index.has_duplicates:
        raise ValueError("дубльовані (stint_code, lap_in_stint) — перевір, що файл від одного запуску")
    common = A.index.intersection(B.index)
    p = A.loc[common, ["race", "Location", "Compound", "Driver", "n_laps", "n_prefix"]].copy()
    for c in ("abs_err", "crps", "err", "cover94", "width94"):
        p[c + "_A"] = A.loc[common, c].to_numpy()
        p[c + "_B"] = B.loc[common, c].to_numpy()
    p = p.reset_index()
    p["h"] = p["lap_in_stint"] - p["n_prefix"] + 1
    p["d_mae"] = p["abs_err_A"] - p["abs_err_B"]
    p["d_crps"] = p["crps_A"] - p["crps_B"]
    sa = stints[stints["model"] == a].set_index("stint_code")
    sb = stints[stints["model"] == b].set_index("stint_code")
    for col, src, name in (("pf_ess_min", sa, "ess_A"), ("pf_ess_min", sb, "ess_B"),
                           ("nu_median_prefix", sa, "nu_A"), ("nu_median_prefix", sb, "nu_B"),
                           ("new_track", sa, "new_track"), ("new_driver", sa, "new_driver")):
        if col in src.columns:
            p[name] = p["stint_code"].map(src[col])
    return p


def per_race(p: pd.DataFrame) -> pd.DataFrame:
    t = p.groupby("race").agg(n_stints=("stint_code", "nunique"), n_laps=("d_mae", "size"),
                              mae_A=("abs_err_A", "mean"), mae_B=("abs_err_B", "mean"), d_mae=("d_mae", "mean"),
                              d_crps=("d_crps", "mean"), bias_A=("err_A", "mean"), bias_B=("err_B", "mean"))
    return t.sort_values("d_mae", ascending=False)


def summarise_by(p: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in p.groupby(col, observed=True, dropna=False, sort=True):
        m, se, G = _cluster_mean(g["d_mae"], g["race"])
        mc, sec, _ = _cluster_mean(g["d_crps"], g["race"])
        rows.append({col: key, "n_laps": len(g), "n_stints": g["stint_code"].nunique(), "n_races": G,
                     "mae_A": g["abs_err_A"].mean(), "mae_B": g["abs_err_B"].mean(), "d_mae": m, "se": se,
                     "d_crps": mc, "se_crps": sec, "bias_A": g["err_A"].mean(), "bias_B": g["err_B"].mean()})
    return pd.DataFrame(rows)


def ess_table(stints: pd.DataFrame) -> pd.DataFrame:
    pf = stints[stints["model"] != OLS]
    if "pf_ess_min" not in pf.columns or pf.empty:
        return pd.DataFrame()
    g = pf.groupby("model")["pf_ess_min"]
    t = g.quantile([0.05, 0.25, 0.5]).unstack()
    t.columns = ["q05", "q25", "q50"]
    t["share_lt_50"] = pf.assign(lt=pf["pf_ess_min"] < 50).groupby("model")["lt"].mean()
    return t


def nu_adaptivity(stints: pd.DataFrame, min_cell: int = 3) -> pd.DataFrame:
    """Наскільки апостеріорне ν (після префікса) слідує за власним OLS-нахилом стінта
    у межах комірки гонка×компаунд: 0 — ігнорує свої дані, 1 — повністю за ними.
    OLS-нахил шумний і містить паливний зсув, але він однаковий для всіх моделей,
    тож порівнювати варто моделі між собою, а не абсолютні значення."""
    if OLS not in set(stints["model"]) or "ols_slope" not in stints.columns:
        return pd.DataFrame()
    ols = stints[stints["model"] == OLS].set_index("stint_code")["ols_slope"]
    rows = []
    for m in sorted(set(stints["model"]) - {OLS}):
        s = stints[stints["model"] == m].set_index("stint_code")
        if "nu_median_prefix" not in s.columns:
            continue
        df = s[["race", "Compound", "nu_median_prefix", "nu_lo", "nu_hi"]].join(ols, how="inner").dropna()
        cell = df["race"].astype(str) + "|" + df["Compound"].astype(str)
        keep = cell.map(cell.value_counts()) >= min_cell
        df, cell = df[keep], cell[keep]
        if len(df) < 5:
            continue
        x = df["ols_slope"] - df.groupby(cell)["ols_slope"].transform("mean")
        y = df["nu_median_prefix"] - df.groupby(cell)["nu_median_prefix"].transform("mean")
        rows.append({"model": m, "n_stints": len(df),
                     "follow_own_slope": float((x * y).sum() / (x * x).sum()),
                     "corr_within_cell": float(np.corrcoef(x, y)[0, 1]),
                     "median_nu": float(df["nu_median_prefix"].median()),
                     "median_rel_width_nu94": float(((df["nu_hi"] - df["nu_lo"]) / df["nu_median_prefix"]).median())})
    return pd.DataFrame(rows)


def add_train_support(p: pd.DataFrame, all_stints: pd.DataFrame, train_years: Sequence[int]) -> pd.DataFrame:
    st = pd.DataFrame(all_stints).reset_index(drop=True)
    missing = {"race", "Location", "Compound"} - set(st.columns)
    if missing:
        print(f"[train support] пропущено: у таблиці стінтів немає колонок {sorted(missing)}")
        return p
    yr = pd.to_numeric(st["Year"], errors="coerce") if "Year" in st.columns else _year_of(st["race"])
    tr = st[yr.isin(list(train_years))]
    if tr.empty:
        print(f"[train support] пропущено: не знайшов стінтів {list(train_years)} (рік беру з Year або з ключа race)")
        return p
    races_per_track = tr.groupby("Location")["race"].nunique()
    stints_per_cell = tr.groupby(["Location", "Compound"]).size()
    p = p.copy()
    p["train_races_track"] = p["Location"].map(races_per_track).fillna(0).astype(int)
    idx = pd.MultiIndex.from_arrays([p["Location"], p["Compound"]])
    p["train_stints_cell"] = stints_per_cell.reindex(idx).fillna(0).astype(int).to_numpy()
    p["train_stints_cell_bin"] = pd.cut(p["train_stints_cell"], [-1, 0, 2, 5, np.inf], labels=["0", "1–2", "3–5", "≥6"])
    return p


# --------------------------------------------------------------------------- driver
def run(forecast_dir, all_stints: Optional[pd.DataFrame] = None, train_years: Sequence[int] = (2022, 2023),
        a: str = MODEL_A, b: str = MODEL_B, out_dir=None, show: bool = True) -> Dict[str, pd.DataFrame]:
    laps, stints, summary = load(forecast_dir)
    p = paired_laps(laps, stints, a, b)
    out: Dict[str, pd.DataFrame] = {}
    say = _show if show else (lambda *args, **kw: None)

    pr = per_race(p)
    out["per_race"] = pr
    chk = f"середнє за гонками MAE: A={pr['mae_A'].mean():.4f}, B={pr['mae_B'].mean():.4f}, A−B={pr['d_mae'].mean():+.4f}"
    if summary is not None:
        s = summary[summary["scheme"] == SCHEME].set_index("model")["MAE_race_mean"]
        if a in s.index and b in s.index:
            chk += f"  (у forecast_summary.csv: A={s[a]:.4f}, B={s[b]:.4f})"
    top3 = pr["d_mae"].nlargest(3).index
    say(f"1. По гонках (fixed origin), A={a}, B={b}; відсортовано за d_mae", pr,
        chk + f"\nA гірша в {int((pr['d_mae'] > 0).sum())} з {len(pr)} гонок; "
              f"середнє A−B без 3 найгірших для A гонок: {pr['d_mae'].drop(top3).mean():+.4f}"
              f"\nрозкид між гонками середнього bias (SD): A={pr['bias_A'].std():.4f}, B={pr['bias_B'].std():.4f} "
              f"(більший у A → спільний для всієї гонки зсув, який A не ловить)")

    out["by_compound"] = summarise_by(p, "Compound")
    say("2. За компаундом (середнє по колах, SE — кластери-гонки)", out["by_compound"])

    p["prefix_bin"] = pd.cut(p["n_prefix"], [0, 8, 14, 24, np.inf], labels=["≤8", "9–14", "15–24", "≥25"])
    out["by_prefix"] = summarise_by(p, "prefix_bin")
    say("3. За довжиною префікса (кіл, на яких калібрується фільтр)", out["by_prefix"])

    p["h_bin"] = pd.cut(p["h"], [0, 1, 2, 3, 5, 8, np.inf], labels=["1", "2", "3", "4–5", "6–8", "≥9"])
    out["by_horizon"] = summarise_by(p, "h_bin")
    say("4. За горизонтом h (кіл після префікса)", out["by_horizon"],
        "Якщо |bias| і d_mae ростуть з h — проблема в нахилі (ν), якщо сталі — в рівні/шумі.")

    for col, title in (("new_track", "5a. Нова траса (немає в train → ефекти з популяції)"),
                       ("new_driver", "5b. Новий пілот")):
        if col in p.columns:
            out[f"by_{col}"] = summarise_by(p, col)
            say(title, out[f"by_{col}"])

    out["ess"] = ess_table(stints)
    say("6. Мінімальний ESS фільтра на префіксі, по моделях", out["ess"],
        "У notebook квантилі ESS змішані по всіх моделях; тут — окремо.")
    if "ess_A" in p.columns and p["ess_A"].notna().any():
        thr = stints.loc[stints["model"] == a, "pf_ess_min"].quantile(0.10)
        p["ess_A_low10"] = p["ess_A"] < thr
        out["by_ess_A"] = summarise_by(p, "ess_A_low10")
        say(f"6b. Стінти з ESS моделі A у нижніх 10% (< {thr:.1f}) проти решти", out["by_ess_A"])

    out["nu_adaptivity"] = nu_adaptivity(stints)
    say("7. Чи слідує ν після префікса за власним нахилом стінта (у межах гонка×компаунд)", out["nu_adaptivity"],
        "follow_own_slope ≈ 0: ν тримається апріорного значення; більше — сильніше підлаштовується під стінт.")
    if {"nu_A", "nu_B"}.issubset(p.columns):
        st_ratio = p.drop_duplicates("stint_code").set_index("stint_code")
        lr = np.log(st_ratio["nu_A"] / st_ratio["nu_B"]).replace([np.inf, -np.inf], np.nan).dropna()
        if lr.nunique() >= 5:
            codes, edges = pd.qcut(lr, 5, labels=False, retbins=True, duplicates="drop")
            labels = [f"Q{i + 1} ({edges[i]:+.2f}…{edges[i + 1]:+.2f})" for i in range(len(edges) - 1)]
            mapped = p["stint_code"].map(codes).map(lambda k: labels[int(k)] if pd.notna(k) else np.nan)
            p["log_nuA_over_nuB_q"] = pd.Categorical(mapped, categories=labels, ordered=True)
            out["by_nu_ratio"] = summarise_by(p, "log_nuA_over_nuB_q")
            say("7b. За квінтилями log(ν_A/ν_B) після префікса", out["by_nu_ratio"],
                "Якщо A програє переважно в крайніх квінтилях — розрив іде від різної оцінки ν.")

    if all_stints is not None:
        p = add_train_support(p, all_stints, train_years)
        if "train_races_track" in p.columns:
            out["by_train_races"] = summarise_by(p, "train_races_track")
            say(f"8a. Скільки гонок цієї траси було в train {list(train_years)}", out["by_train_races"])
            out["by_train_cell"] = summarise_by(p, "train_stints_cell_bin")
            say("8b. Скільки стінтів комірки траса×компаунд було в train", out["by_train_cell"],
                "Для комірки з 0 стінтів primary бере I[q,c], який задають zero-sum обмеження, а не N(0, σ_I²).")

    od = Path(out_dir) if out_dir is not None else Path.cwd() / "forecast_diagnostics"
    od.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        if df is not None and len(df):
            df.to_csv(od / f"{k}.csv", index=(k in ("per_race", "ess")))
    p.to_csv(od / "paired_laps.csv", index=False)
    if show:
        print(f"\nтаблиці збережено в {od}")
    out["paired_laps"] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forecast_dir", help="папка з forecast_laps.csv і forecast_stints.csv")
    ap.add_argument("--all-stints", help="CSV з усіма стінтами (колонки race, Location, Compound[, Year])")
    ap.add_argument("--train-years", nargs="+", type=int, default=[2022, 2023])
    ap.add_argument("--a", default=MODEL_A)
    ap.add_argument("--b", default=MODEL_B)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    st = pd.read_csv(args.all_stints) if args.all_stints else None
    run(args.forecast_dir, all_stints=st, train_years=args.train_years, a=args.a, b=args.b, out_dir=args.out)


if __name__ == "__main__":
    main()
