"""Paper-ready tables (Ukrainian headers, decimal comma, intervals as «оцінка [нижня; верхня]»).

`build(bundle, out_dir)` turns the DataFrames that the export stage has just computed into the tables of
the manuscript and writes
    paper/tabN_*.csv|md      one file pair per manuscript table
    paper/paper_numbers.json every number quoted in the running text, unrounded, with its source table
    paper/MISSING.md         what could not be built and which experiment has to be run for it

Nothing here computes a result: every cell is a formatted copy of a value from `tables/`, `forecast/`,
`loo/` or `diagnostics/`.  A table whose inputs do not exist is not written; it is listed in MISSING.md,
so that a placeholder in the manuscript can never be mistaken for a result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SYMBOL = {"mu_log_nu": "μ_ν", "mu_nu_add": "μ_ν (с/коло)", "sigma_track": "σ_T", "sigma_track_add": "σ_T (с/коло)", "sigma_interaction": "σ_I", "sigma_interaction_add": "σ_I (с/коло)",
          "sigma_stint": "σ_U", "sigma_stint_add": "σ_U (с/коло)", "sigma_team": "σ_G", "sigma_team_add": "σ_G (с/коло)", "sigma_driver": "σ_D", "sigma_driver_add": "σ_D (с/коло)",
          "sigma_race": "σ_R", "sigma_race_add": "σ_R (с/коло)", "sigma_compound": "σ_C", "baseline_mean": "β₀", "sigma_stint_base": "σ_β", "mu_log_gamma": "μ_γ",
          "sigma_log_gamma": "σ_γ", "sigma_eta": "σ_η", "sigma_eps_core": "σ_core", "lambda_skew": "λ", "nu_df": "η", "compound_effect": "C_c", "compound_base_effect": "b_c",
          "track_effect": "T_q", "gamma_track": "γ_q", "team_effect": "G_g", "driver_effect": "D_d", "interaction_effect": "I_qc", "race_effect": "R_r"}
COMPOUND_UK = {"SOFT": "Soft", "MEDIUM": "Medium", "HARD": "Hard"}


def num(x, nd=3, comma=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{float(x):.{nd}f}".replace("-", "−")
    return s.replace(".", ",") if comma else s


def iv(med, lo, hi, nd=3) -> str:
    if med is None or not np.isfinite(med):
        return "—"
    return f"{num(med, nd)} [{num(lo, nd)}; {num(hi, nd)}]"


class _Out:
    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.written: Dict[str, str] = {}; self.missing: List[str] = []; self.numbers: Dict[str, Any] = {}

    def table(self, name: str, df: pd.DataFrame, caption: str, note: str = ""):
        df.to_csv(self.dir / f"{name}.csv", index=False)
        try:
            md = df.to_markdown(index=False)
        except Exception:
            md = df.to_string(index=False)
        (self.dir / f"{name}.md").write_text(f"**{caption}**\n\n{md}\n" + (f"\n{note}\n" if note else ""), encoding="utf-8")
        self.written[name] = str(self.dir / f"{name}.csv")

    def miss(self, what: str, why: str):
        self.missing.append(f"- **{what}** — {why}")

    def n(self, key: str, value, source: str):
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        self.numbers[key] = {"value": value, "source": source}


def build(b: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """b: bundle from Pipeline.stage_export —
    primary (name), prob, priors (df), hyper (df|None), gates (df|None), R (results dict|None), comparison (df|None),
    loo (df|None), forecast (df|None), forecast_pairs (df|None), forecast_extra ({tag: (summary, pairs)}), forecast_info (dict|None),
    hierarchy_ablation (df|None), ppc (df|None), pit_tails (dict|None), recovery_summary ({label: df}), recovery_coverage ({label: dict}), recovery_source ({label: "run"|"results_sandbox"}),
    experiments (df), counts (dict|None),
    cleaning (df|None), ols (dict|None), data_sensitivity (df|None)."""
    o = _Out(out_dir)
    P = int(round(100 * b.get("prob", 0.94)))
    prim = b.get("primary", "primary")

    # ---- Table 1: priors ------------------------------------------------------------------------
    pr = b.get("priors")
    if pr is not None and len(pr):
        t = pr[["symbol", "role", "distribution", "units", "block"]].rename(columns={"symbol": "Параметр", "role": "Зміст", "distribution": "Апріорний розподіл", "units": "Одиниці", "block": "Блок"})
        o.table("tab1_priors", t, f"Таблиця 1 — Апріорні розподіли моделі «{prim}» (згенеровано з коду моделі)")

    # ---- Table 3: MCMC diagnostics of the hyper-parameters ---------------------------------------
    hy = b.get("hyper")
    if hy is not None and len(hy):
        rows = []
        for _, r in hy.iterrows():
            scalar = r["kind"] == "scalar"
            rows.append({"Параметр": SYMBOL.get(r["parameter"], r["parameter"]) + ("" if scalar else f" ({int(r['n_elements'])} ел., найгірший)"),
                         "Середнє": num(r["mean"], 4) if scalar else "—", "SD": num(r["sd"], 4) if scalar else "—", f"{P} % ETI": f"[{num(r['lo'], 4)}; {num(r['hi'], 4)}]" if scalar else "—",
                         "R̂": num(r["rhat"], 4), "ESS bulk": num(r["ess_bulk"], 0), "ESS tail": num(r["ess_tail"], 0), "MCSE/SD": num(r["mcse_rel"], 3)})
            o.n(f"mcmc.{r['parameter']}.rhat", r["rhat"], "tab3_mcmc"); o.n(f"mcmc.{r['parameter']}.ess_bulk", r["ess_bulk"], "tab3_mcmc"); o.n(f"mcmc.{r['parameter']}.ess_tail", r["ess_tail"], "tab3_mcmc")
        o.table("tab3_mcmc", pd.DataFrame(rows), f"Таблиця 3 — Апостеріорні оцінки та діагностика MCMC гіперпараметрів моделі «{prim}»",
                "R̂ — рангово-нормалізований розщеплений; ESS — ефективний розмір вибірки (основна частина / хвости); для векторних параметрів наведено найгірший елемент.")
    else:
        o.miss("Таблиця 3 (R̂, ESS гіперпараметрів)", f"немає diagnostics/hyper_{prim}.csv: завершіть fit `{prim}` або виконайте REPORT_ONLY")
    g = b.get("gates")
    if g is not None and len(g):
        t = pd.DataFrame({"Модель": g["model"], "Ланцюги × ітерації": [f"{int(c)} × {int(d)}" for c, d in zip(g["chains"], g["draws"])], "R̂ max": [num(v, 4) for v in g["worst_rhat"]],
                          "ESS bulk min": [num(v, 0) for v in g["min_ess_bulk"]], "ESS tail min": [num(v, 0) for v in g["min_ess_tail"]], "MCSE/SD max": [num(v, 3) for v in g["max_mcse_rel"]],
                          "Дивергенції": g["divergences"].astype(int), "E-BFMI min": [num(v, 3) for v in g["ebfmi_min"]],
                          "Час вибірки, год": [num(v / 3600.0, 1) if v == v and v is not None else "—" for v in g["sampling_wall_s"]],
                          "Критерії": ["пройдено" if not f or f != f else "не пройдено: " + str(f) for f in g["failures"]]})
        o.table("tab3b_gates", t, "Таблиця 3б — Операційні критерії діагностики MCMC за всіма завершеними моделями")

    # ---- Tables 4–5: main results ----------------------------------------------------------------
    R = b.get("R")
    if R is not None:
        gr = R["global_rate"]
        for k in ("median", "lo", "hi"):
            o.n(f"global_rate.{k}", gr[k], f"global_rate ({prim})")
        comp = R["compound"]
        rows = []
        for _, r in comp.iterrows():
            rows.append({"Компаунд": COMPOUND_UK.get(r["compound"], r["compound"]), "Відрізків": int(r["n_stints"]), "Кіл": int(r["n_laps"]),
                         "Множник exp(C_c)": iv(r["multiplier_median"], r["multiplier_lo"], r["multiplier_hi"], 3), "Швидкість, с/коло": iv(r["rate_median_s_per_lap"], r["rate_lo"], r["rate_hi"], 4)})
            for k in ("multiplier_median", "multiplier_lo", "multiplier_hi", "rate_median_s_per_lap", "rate_lo", "rate_hi"):
                o.n(f"compound.{r['compound']}.{k}", r[k], f"compound_{prim}")
        rows.append({"Компаунд": "Загальний рівень exp(μ_ν)", "Відрізків": int(comp["n_stints"].sum()), "Кіл": int(comp["n_laps"].sum()), "Множник exp(C_c)": "1", "Швидкість, с/коло": iv(gr["median"], gr["lo"], gr["hi"], 4)})
        o.table("tab4_compounds", pd.DataFrame(rows), f"Таблиця 4 — Швидкість деградації за компаундами (медіана та {P} % ETI)")
        cc = R["compound_contrasts"]
        t = pd.DataFrame({"Контраст": cc["contrast"], "Різниця швидкостей, с/коло": [iv(a, b_, c, 4) for a, b_, c in zip(cc["rate_diff_median_s_per_lap"], cc["rate_diff_lo"], cc["rate_diff_hi"])],
                          "Відношення швидкостей": [iv(a, b_, c, 3) for a, b_, c in zip(cc["ratio_median"], cc["ratio_lo"], cc["ratio_hi"])],
                          "P(різниця > 0)": [num(v, 4) for v in cc["P(diff>0)"]], "MCSE або межа 3/ESS": [num(v, 4) for v in cc["P_mcse_or_bound"]]})
        o.table("tab4b_contrasts", t, "Таблиця 4б — Попарні контрасти компаундів зі спільних апостеріорних реалізацій")
        od = R["order_probabilities"]
        if len(od):
            gl = od[od["scope"].str.startswith("global")]
            for _, r in gl.iterrows():
                o.n(f"order.{r['event']}", r["probability"], f"order_probabilities_{prim}")
            tr = od[~od["scope"].str.startswith("global")].copy()
            if len(tr):
                tr["Траса"] = tr["scope"].str.replace("track ", "", regex=False)
                t = pd.DataFrame({"Траса": tr["Траса"], "Відрізків S/M/H": tr["n_stints_S/M/H"], "P(S > M > H)": [num(v, 3) for v in tr["probability"]], "P(S > H)": [num(v, 3) for v in tr["P(SOFT>HARD)"]]})
                o.table("tabS_order_by_track", t.sort_values("Траса"), "Таблиця Д1 — Ймовірність порядку швидкостей деградації за трасами (повні швидкості з взаємодією)")
                o.n("order.tracks_ge_0.9", int((tr["probability"] >= 0.9).sum()), f"order_probabilities_{prim}"); o.n("order.n_tracks", len(tr), f"order_probabilities_{prim}")
        sc = R["scales"]
        comps = [c for c in dict.fromkeys(sc["component"]) if c not in ("разом", "2×коваріації (перехресний дизайн)")]
        rows = []
        for c in comps:
            s = sc[sc["component"] == c]
            h = s[s["kind"].str.startswith("hyper")]; r_ = s[s["kind"].str.startswith("realised")]; v = s[s["kind"].str.startswith("variance")]
            rows.append({"Компонент": c, "σ (гіперпараметр)": iv(h["median"].iloc[0], h["lo"].iloc[0], h["hi"].iloc[0], 3) if len(h) else "фіксована шкала",
                         "Реалізований SD ефектів": iv(r_["median"].iloc[0], r_["lo"].iloc[0], r_["hi"].iloc[0], 3) if len(r_) else "—",
                         "Частка дисперсії": iv(v["median"].iloc[0], v["lo"].iloc[0], v["hi"].iloc[0], 3) if len(v) else "—", "Одиниці σ": h["units"].iloc[0] if len(h) and "units" in h else ""})
            if len(v):
                o.n(f"variance_share.{c}", v["median"].iloc[0], f"scales_{prim}")
        for label in ("2×коваріації (перехресний дизайн)", "разом"):
            s = sc[sc["component"] == label]
            if len(s):
                rows.append({"Компонент": label, "σ (гіперпараметр)": "—", "Реалізований SD ефектів": iv(s["median"].iloc[0], s["lo"].iloc[0], s["hi"].iloc[0], 3) if label == "разом" else "—",
                             "Частка дисперсії": iv(s["median"].iloc[0], s["lo"].iloc[0], s["hi"].iloc[0], 3) if label != "разом" else "1", "Одиниці σ": ""})
        o.table("tab5_scales", pd.DataFrame(rows), f"Таблиця 5 — Джерела неоднорідності швидкості деградації: масштаби та частки дисперсії лінійного предиктора (медіана та {P} % ETI)")
        rs = R["residual"]
        if len(rs):
            t = pd.DataFrame({"Параметр": rs["parameter"], "Медіана": [num(v, 4) for v in rs["median"]], f"{P} % ETI": [f"[{num(a, 4)}; {num(c, 4)}]" if a == a else "—" for a, c in zip(rs["lo"], rs["hi"])]})
            o.table("tab5b_noise", t, "Таблиця 5б — Параметри похибок спостережень, процесного шуму, базових рівнів і паливного коефіцієнта")
        sg = R.get("sigma_cmp")
        if sg is not None and len(sg):
            r = sg.iloc[0]
            o.n("sigma_G_minus_sigma_D.median", r["median"], f"sigma_comparison_{prim}"); o.n("sigma_G_minus_sigma_D.lo", r["lo"], f"sigma_comparison_{prim}")
            o.n("sigma_G_minus_sigma_D.hi", r["hi"], f"sigma_comparison_{prim}"); o.n("P(sigma_G>sigma_D)", r["P(>0)"], f"sigma_comparison_{prim}")
        tk = R.get("tracks")
        if tk is not None and len(tk):
            t = pd.DataFrame({"Траса": tk["track"], "Гонок": tk["n_races"], "Відрізків": tk["n_stints"], "Швидкість, с/коло": [iv(a, b_, c, 4) for a, b_, c in zip(tk["rate_median_s_per_lap"], tk["rate_lo"], tk["rate_hi"])],
                              "Відносна швидкість, % часу кола": [iv(a, b_, c, 3) for a, b_, c in zip(tk["relative_rate_pct_per_lap"], tk["relative_rate_lo"], tk["relative_rate_hi"])]})
            if "gamma_median_s_per_kg" in tk:
                t["γ_q, с/кг"] = [iv(a, b_, c, 4) for a, b_, c in zip(tk["gamma_median_s_per_kg"], tk["gamma_lo"], tk["gamma_hi"])]
            o.table("tabS_tracks", t.sort_values("Траса"), "Таблиця Д2 — Трасові швидкості деградації (абсолютні та відносні) і паливно-трасовий коефіцієнт")
        sn = R.get("sign")
        if sn is not None and len(sn):
            t = sn.rename(columns={"level": "Рівень", "quantity": "Величина", "value": "Значення", "n": "n", "link": "Зв'язок", "by_construction": "За побудовою", "note": "Примітка"})
            o.table("tabS_sign", t, "Таблиця Д3 — Знак швидкості деградації за рівнями (частка апостеріорної маси)")
    else:
        o.miss("Таблиці 4–5 (основні результати)", f"fit `{prim}` не завершено")

    # ---- Table 7: model comparison ---------------------------------------------------------------
    cmp_ = b.get("comparison")
    if cmp_ is not None and len(cmp_):
        loo = b.get("loo")
        loo_ix = loo.set_index("model") if loo is not None and len(loo) else None
        rows = []
        for _, r in cmp_.iterrows():
            row = {"Модель": r["model"], "Дані": r.get("data", ""), "Відмінність від основної": r.get("differs", ""),
                   "R̂ max": num(r.get("worst_rhat"), 4), "ESS min": num(r.get("min_ess"), 0), "E-BFMI min": num(r.get("ebfmi_min"), 3),
                   "Загальна швидкість, с/коло": iv(r["global_rate"], r["global_lo"], r["global_hi"], 4),
                   "Soft/Hard": iv(r.get("ratio_SH"), r.get("ratio_SH_lo"), r.get("ratio_SH_hi"), 3) if "ratio_SH" in r else "—",
                   "σ_T": num(r.get("hyperSD_траса"), 3), "σ_U": num(r.get("hyperSD_відрізок (залишок)"), 3), "σ_core": num(r.get("sigma_eps_core"), 3), "η": num(r.get("nu_df"), 3),
                   "σ_η": num(r.get("sigma_eta"), 3),
                   # realised SDs of the fitted group effects: defined with or without learned scales, hence comparable between
                   # the hierarchical model and a specification with fixed scales (where the σ columns are empty by definition)
                   "SD ефектів трас (реаліз.)": num(r.get("realSD_траса"), 3), "SD ефектів пілотів (реаліз.)": num(r.get("realSD_пілот"), 3),
                   "SD ефектів конструкторів (реаліз.)": num(r.get("realSD_конструктор"), 3), "SD взаємодії (реаліз.)": num(r.get("realSD_траса×компаунд"), 3),
                   "Масштаби T, D, G, I": {"hier": "оцінюються", "fixed": "фіксовані"}.get(str(r.get("group_scale", "hier")), str(r.get("group_scale", "")))}
            if loo_ix is not None and r["model"] in loo_ix.index:
                L = loo_ix.loc[r["model"]]
                row["elpd_LOO на коло"] = num(L["elpd_loo_per_lap"], 4)
                row["Δelpd_LOO до основної (SE за відрізками)"] = f"{num(L['d_elpd_loo_vs_ref'], 1)} ({num(L['se_d_elpd_loo_stint_clustered'], 1)})" if "d_elpd_loo_vs_ref" in L and L["d_elpd_loo_vs_ref"] == L["d_elpd_loo_vs_ref"] else "—"
                row["Частка k̂ > 0,7"] = num(L["share_k_gt_0.7"], 3)
            rows.append(row)
            for k in ("global_rate", "global_lo", "global_hi", "ratio_SH", "ratio_SH_lo", "ratio_SH_hi"):
                if k in r:
                    o.n(f"model.{r['model']}.{k}", r[k], "model_comparison")
        o.table("tab7_models", pd.DataFrame(rows), f"Таблиця 7 — Порівняння специфікацій моделі (медіана та {P} % ETI)",
                "Δelpd порівнюються лише для моделей на тих самих колах; PSIS-LOO умовний на латентні стани, тому чутливий насамперед до моделі похибок. "
                "«Реаліз.» — стандартне відхилення оцінених ефектів (медіана): його можна порівнювати між моделями з навченими й фіксованими масштабами; "
                "стовпець σ_T порожній там, де σ_T не оцінюється.")
    else:
        o.miss("Таблиця 7 (порівняння моделей)", "немає завершених fits")

    # ---- Table 8: forecast -----------------------------------------------------------------------
    def forecast_tables(summ, pairs, suffix, title):
        f = summ[summ["scheme"] == "fixed_origin"]
        t = pd.DataFrame({"Модель": f["model"], "Кіл": f["n_laps"].astype(int), "Гонок": f["n_races"].astype(int), "MAE, с": [num(v, 3) for v in f["MAE"]], "RMSE, с": [num(v, 3) for v in f["RMSE_pooled"]],
                          "CRPS, с": [num(v, 3) for v in f["CRPS"]], "LPD": [num(v, 3) for v in f["LPD"]] if "LPD" in f else "—",
                          f"Покриття {P} %": [num(v, 3) for v in f["coverage94"]], "Покриття 50 %": [num(v, 3) for v in f["coverage50"]]})
        o.table(f"tab8_forecast{suffix}", t, title)
        for _, r in f.iterrows():
            for k in ("MAE", "RMSE_pooled", "CRPS", "LPD", "coverage94", "coverage50"):
                if k in r:
                    o.n(f"forecast{suffix}.{r['model']}.{k}", r[k], f"forecast_summary{suffix}")
        rl = summ[summ["scheme"] == "rolling_one_step"]
        if len(rl):
            t = pd.DataFrame({"Модель": rl["model"], "MAE, с": [num(v, 3) for v in rl["MAE"]], "RMSE, с": [num(v, 3) for v in rl["RMSE_pooled"]], "CRPS, с": [num(v, 3) for v in rl["CRPS"]],
                              "LPD": [num(v, 3) for v in rl["LPD"]] if "LPD" in rl else "—", f"Покриття {P} %": [num(v, 3) for v in rl["coverage94"]]})
            o.table(f"tab8c_forecast_rolling{suffix}", t, title.replace("фіксований початок", "ковзний прогноз на одне коло"))
        if pairs is not None and len(pairs):
            lab = {"mae": "MAE, с", "crps": "CRPS, с", "lpd": "LPD"}
            t = pd.DataFrame({"Модель a": pairs["model_a"], "Модель b": pairs["model_b"], "Оцінка": [lab.get(m, m) for m in pairs["metric"]], "Гонок": pairs["n_races"].astype(int),
                              "Середня різниця a − b": [num(v, 4) for v in pairs["mean_diff_a_minus_b"]], "SE": [num(v, 4) for v in pairs["se_race"]],
                              f"{P} % ДІ (t)": [f"[{num(a, 4)}; {num(c, 4)}]" for a, c in zip(pairs["ci94_lo"], pairs["ci94_hi"])],
                              "Частка гонок, де a краща": [num(v, 2) for v in pairs["share_races_a_better"]]})
            o.table(f"tab8b_forecast_pairs{suffix}", t, f"Таблиця 8б — Парні різниці прогнозних оцінок за гонками{' (' + suffix.strip('_') + ')' if suffix else ''}",
                    "MAE і CRPS — втрати (менше краще); LPD — логарифмічна предиктивна щільність (більше краще). Одиниця вибірки — гонка.")
            for _, r in pairs.iterrows():
                o.n(f"forecast_pair{suffix}.{r['model_a']}-{r['model_b']}.{r['metric']}.diff", r["mean_diff_a_minus_b"], f"forecast_paired_differences{suffix}")
                o.n(f"forecast_pair{suffix}.{r['model_a']}-{r['model_b']}.{r['metric']}.se", r["se_race"], f"forecast_paired_differences{suffix}")

    fs = b.get("forecast")
    if fs is not None and len(fs):
        forecast_tables(fs, b.get("forecast_pairs"), "", "Таблиця 8 — Прогноз останніх кіл відрізків тестового сезону (фіксований початок)")
        for tag, (s2, p2) in (b.get("forecast_extra") or {}).items():
            if s2 is not None and len(s2):
                forecast_tables(s2, p2, tag, f"Таблиця 8 ({tag.strip('_')}) — Прогноз (фіксований початок), додаткова схема")
        ha = b.get("hierarchy_ablation")
        if ha is not None and len(ha):
            lab = {"mae": "MAE, с", "crps": "CRPS, с", "lpd": "LPD"}
            t = pd.DataFrame({"Схема": ha["scheme"], "Відрізки": ha["stints"], "Порівняння (перша − друга)": ha["comparison"], "Оцінка": [lab.get(m, m) for m in ha["metric"]],
                              "Гонок": ha["n_races"].astype(int), "Середня різниця": [num(v, 4) for v in ha["mean_diff_a_minus_b"]], "SE": [num(v, 4) for v in ha["se_race"]],
                              f"{P} % ДІ (t)": [f"[{num(a, 4)}; {num(c, 4)}]" for a, c in zip(ha["ci94_lo"], ha["ci94_hi"])],
                              "Частка гонок, де перша краща": [num(v, 2) for v in ha["share_races_a_better"]], "Що дозволяє сказати інтервал": ha["reading"]})
            o.table("tab8d_hierarchy_ablation", t, "Таблиця 8г — Внесок групових ефектів і навчання їхніх масштабів у прогноз (парні різниці за гонками)",
                    "pooled_train — без T, D, G, I; unpooled_train — ті самі ефекти з фіксованими апріорними масштабами (σ_T, σ_D, σ_G, σ_I не оцінюються); primary_train — масштаби оцінюються. "
                    "MAE і CRPS — втрати (менше краще); LPD — логарифмічна предиктивна щільність (більше краще). Інтервал, що містить нуль, означає лише, що різницю не підтверджено.")
            for _, r in ha.iterrows():
                o.n(f"hierarchy_ablation.{r['scheme']}.{r['stints']}.{r['model_a']}-{r['model_b']}.{r['metric']}.diff", r["mean_diff_a_minus_b"], "forecast_hierarchy_ablation")
        fi = b.get("forecast_info") or {}
        for k in ("n_test_stints", "n_target_laps", "n_target_laps_strict", "n_test_stints_strict", "train_offset_full_minus_q1_s"):
            if k in fi:
                o.n(f"forecast.{k}", fi[k], "forecast_info.json")
        if fi.get("models_not_run"):
            o.miss("Рядки таблиці 8", "навчальні fits не завершено: " + ", ".join(fi["models_not_run"]))
    else:
        o.miss("Таблиця 8 (прогноз)", "етап forecast не виконано (потрібні завершені *_train fits)")

    # ---- PPC -------------------------------------------------------------------------------------
    ppc = b.get("ppc")
    if ppc is not None and len(ppc):
        tt = ppc[ppc["replicates"] == "truncated_like_data"]
        name = {"q05": "квантиль 0,05, с", "q25": "квантиль 0,25, с", "q50": "медіана, с", "q75": "квантиль 0,75, с", "q95": "квантиль 0,95, с", "q99": "квантиль 0,99, с", "iqr": "IQR, с", "mad": "MAD, с",
                "p_gt_2": "P(y > 2 с)", "p_gt_3": "P(y > 3 с)", "p_gt_5": "P(y > 5 с)", "p_lt_m2": "P(y < −2 с)", "skew_raw_unreliable": "вибіркова скошеність (ненадійна при η → 2)", "mean": "середнє, с", "max": "максимум, с"}
        t = pd.DataFrame({"Статистика": [name.get(s, s) for s in tt["statistic"]], "Спостережено": [num(v, 3) for v in tt["observed"]], "Репліки (середнє)": [num(v, 3) for v in tt["rep_mean"]],
                          "Репліки, 90 % інтервал": [f"[{num(a, 3)}; {num(c, 3)}]" for a, c in zip(tt["rep_q05"], tt["rep_q95"])], "p (двобічне)": [num(v, 3) for v in tt["bayes_p_two_sided"]]})
        o.table("tab6_ppc", t, f"Таблиця 6 — Умовна апостеріорна предиктивна перевірка моделі «{prim}» (репліки усічено тим самим правилом, що й дані)")
        for k, v in (b.get("pit_tails") or {}).items():
            o.n(f"ppc.{k}", v, f"ppc_conditional_{prim}")
    else:
        o.miss("Таблиця 6 (предиктивні перевірки)", f"етап ppc_{prim} не виконано")

    # ---- recovery --------------------------------------------------------------------------------
    for label, rs_ in (b.get("recovery_summary") or {}).items():
        if rs_ is None or not len(rs_):
            continue
        keep = rs_[rs_["parameter"].isin(["rate_global (s/lap)", "rate_SOFT (s/lap)", "rate_MEDIUM (s/lap)", "rate_HARD (s/lap)", "rate_SOFT_minus_HARD (s/lap)", "ratio_SOFT_over_HARD", "gamma_track",
                                          "mu_log_nu", "mu_nu_add", "sigma_track", "sigma_interaction", "sigma_stint", "sigma_team", "sigma_driver", "sigma_eta", "sigma_eps_core", "lambda_skew", "nu_df",
                                          "track_effect", "interaction_effect", "compound_effect", "nu_stint (per-stint rates)", "sigma_race", "sigma_stint_add", "sigma_track_add"])]
        t = pd.DataFrame({"Сценарій": keep["scenario"], "Модель": keep["variant"], "Коректно специфікована": ["так" if w else "ні" for w in keep["well_specified"]],
                          "Параметр": [SYMBOL.get(p, p) for p in keep["parameter"]], "n": keep["n"].astype(int), "Реплік": keep["n_replicates"].astype(int),
                          "Покриття": [num(v, 3) for v in keep["coverage"]], "SE покриття": [num(v, 3) for v in keep["coverage_se"]], "Середній z": [num(v, 2) for v in keep["mean_z"]],
                          "RMS z": [num(v, 2) for v in keep["rms_z"]], "Зміщення": [num(v, 4) for v in keep["mean_bias"]], "RMSE": [num(v, 4) for v in keep["rmse"]]})
        shipped = (b.get("recovery_source") or {}).get(label) == "results_sandbox"
        o.table(f"tab9_recovery{label}", t, f"Таблиця 9 — Відновлення параметрів на синтетичних даних{' (' + label.strip('_') + ')' if label else ''}",
                f"Номінальне покриття {P} %; z = (істина − апостеріорне середнє)/SD; «ні» — модель навмисно не містить породжувального процесу."
                + (" Джерело: results_sandbox/ (скорочені ланцюги, виконано під час редакції v1.1; власного дослідження в outputs не знайдено)." if shipped else ""))
        cov = (b.get("recovery_coverage") or {}).get(label) or {}
        for k in ("share", "binomial_se", "n_elements", "n_fits"):
            if k in cov:
                o.n(f"recovery{label}.pooled_coverage.{k}", cov[k], f"synthetic_recovery_coverage{label}.json")
    if not (b.get("recovery_summary") or {}):
        o.miss("Таблиця 9 (відновлення параметрів)", "етап recovery не виконано")

    # ---- experiment registry ---------------------------------------------------------------------
    ex = b.get("experiments")
    if ex is not None and len(ex):
        t = pd.DataFrame({"Експеримент": ex["experiment"], "Пріоритет": ex["пріоритет"], "Зауваження": ex["reviewer_points"], "Відмінність від основної моделі": ex["differs_from_primary"], "Стан": ex["статус"]})
        o.table("tab2_experiments", t, "Таблиця 2 — Реєстр специфікацій та їхній фактичний стан")
        for _, r in ex.iterrows():
            if not r["completed"] and r["priority"] in ("mandatory", "recommended"):
                o.miss(f"Результати експерименту `{r['experiment']}` ({r['пріоритет']}; {r['reviewer_points']})", str(r["статус"]))

    # ---- data ------------------------------------------------------------------------------------
    for k, v in (b.get("counts") or {}).items():
        o.n(f"data.{k}", v, "data_counts.csv")
    cl = b.get("cleaning")
    if cl is not None and len(cl):
        step_uk = {"valid lap time": "чинний час кола", "not Deleted": "коло не анульовано", "no in-/out-laps": "без кіл заїзду на піт-лейн і виїзду з нього", "drop race lap 1": "без першого кола гонки",
                   "green-flag only (TrackStatus==1)": "лише зелений прапор", "valid TyreLife & Stint": "чинні вік шини та номер відрізка"}

        def uk(s):
            s = str(s)
            if s in step_uk:
                return step_uk[s]
            if s.startswith("dry compound"):
                return "лише сухі компаунди (Soft, Medium, Hard)"
            if s.startswith("<="):
                return f"час кола ≤ {s.split()[1]} × найшвидше чисте коло гонки"
            if s.startswith("stint >="):
                return f"відрізки з ≥ {s.split()[2]} збережених кіл"
            if s.startswith("years in"):
                return "сезони " + s[len("years in "):]
            if s.startswith("tracks in"):
                return "траси " + s[len("tracks in "):]
            return s
        t = pd.DataFrame({"№": cl["order"], "Крок": [uk(s) for s in cl["step"]], "Залишилося кіл": cl["rows"].astype(int), "Вилучено": cl["removed"].astype(int), "Частка від початкових": [num(v, 4) for v in cl["retained_share"]]})
        o.table("tabS_cleaning", t, "Таблиця Д4 — Послідовність очищення даних")
    ds_ = b.get("data_sensitivity")
    if ds_ is not None and len(ds_):
        o.table("tabS_data_sensitivity", ds_, "Таблиця Д5 — Чутливість обсягу даних і МНК-нахилів до порога очищення та мінімальної довжини відрізка (без повторного оцінювання моделі)")
    for k, v in (b.get("ols") or {}).items():
        if isinstance(v, (int, float)):
            o.n(f"ols.{k}", v, "ols baseline")

    # the directory holds only what this function writes: tables left by an earlier call that are no longer produced
    # (an experiment was removed from the outputs, a table was renamed) would be read as current results
    for old_file in list(o.dir.glob("tab*.csv")) + list(o.dir.glob("tab*.md")):
        if old_file.stem not in o.written:
            old_file.unlink()
    with open(o.dir / "paper_numbers.json", "w", encoding="utf-8") as f:
        json.dump(o.numbers, f, ensure_ascii=False, indent=1, default=float)
    (o.dir / "MISSING.md").write_text("# Чого бракує для статті (згенеровано автоматично)\n\n" + ("\n".join(o.missing) if o.missing else "Усі таблиці побудовано; незапущених обов'язкових або рекомендованих експериментів немає.") + "\n", encoding="utf-8")
    return {"tables": o.written, "missing": o.missing, "numbers_json": str(o.dir / "paper_numbers.json"), "missing_md": str(o.dir / "MISSING.md")}
