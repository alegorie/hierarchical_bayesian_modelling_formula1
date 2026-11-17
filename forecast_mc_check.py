"""Перевірка шуму фільтра частинок у прогнозі primary_train vs pooled_train.

Запуск у ноутбуці ПІСЛЯ комірки конфігурації (потрібні cfg і pl):
    %run -i forecast_mc_check.py

Що робить: повторює лише етап прогнозу (так само, як pl.stage_forecast), але
  * пише в окрему папку outputs/<mode>/forecast_mc_check/<прогін>/;
  * не чіпає реєстр стадій (stages.json), fits і outputs/<mode>/forecast/;
  * змінює лише кількість частинок, seed і, за бажанням, вмикає спільні
    випадкові числа для обох моделей на кожному стінті (common random numbers).
Прогін "orig" має відтворити основний результат один в один — це перевірка, що все зібрано так само.
Після вимкнення світла просто запусти знову: готові прогони пропускаються.
"""
import dataclasses
import json
import time

import numpy as np
import pandas as pd
from IPython.display import display

import f1deg.forecast as F
from f1deg.config import derive_seed

try:
    cfg, pl  # noqa: B018
except NameError as err:
    raise RuntimeError("запускай із ноутбука після комірки конфігурації: %run -i forecast_mc_check.py") from err

MODELS = ["primary_train", "pooled_train"]
K0 = int(cfg.forecast.particles)
RUNS = [  # назва, частинок, seed (None = як в основному прогоні), спільні випадкові числа для моделей
    ("orig", K0, None, False),
    ("seed1", K0, 1, False),
    ("seed2", K0, 2, False),
    ("crn1", K0, 1, True),
    ("crn2", K0, 2, True),
    ("crn1_x4", 4 * K0, 1, True),
    ("crn2_x4", 4 * K0, 2, True),
]
MC_DIR = cfg.out_dir / "forecast_mc_check"

# --- те саме, що збирає pl.stage_forecast(), без реєстру стадій
exps = {e.name: e for e in cfg.experiments}
train_ds, test_ds = pl.dataset({"years": "train"}), pl.dataset({"years": "test"})
assert list(train_ds.levels["compound"]) == list(test_ds.levels["compound"]), "коди компаундів у train і test різні!"
models = {n: (pl.runner(exps[n]).load_posterior(verify=False), exps[n].model, train_ds.levels) for n in MODELS}
offset = float((train_ds.races["race_median_full"] - train_ds.races["race_median_q1"]).mean())
gamma_ext = cfg.primary.gamma_fixed_value

_derive_orig = F.derive_seed


def _derive_crn(seed, *parts):
    """Для фільтра прибирає назву моделі з seed: на стінті обидві моделі беруть однакові випадкові числа."""
    if len(parts) >= 3 and parts[0] == "pf":
        parts = (parts[0],) + tuple(parts[2:])
    return _derive_orig(seed, *parts)


print(f"частинок в основному прогоні: {K0}; прогонів: {len(RUNS)}; папка: {MC_DIR}")
for name, K, s, crn in RUNS:
    out = MC_DIR / name
    if (out / "forecast_summary.csv").exists():
        print(f"{name}: вже є, пропускаю")
        continue
    out.mkdir(parents=True, exist_ok=True)
    (out / "mc_settings.json").write_text(json.dumps({"particles": int(K), "seed_tag": s, "crn": crn, "models": MODELS}), encoding="utf-8")
    fc = dataclasses.replace(cfg.forecast, particles=int(K))
    seed = derive_seed(cfg.seed, "forecast") if s is None else derive_seed(cfg.seed, "forecast_mc", str(s))
    F.derive_seed = _derive_crn if crn else _derive_orig
    t0 = time.time()
    try:
        F.evaluate_models(test_ds, models, fc, gamma_ext, offset, out, seed, log=lambda m: None)
    finally:
        F.derive_seed = _derive_orig
    print(f"{name}: готово за {time.time() - t0:.0f} с")

# --- зведення: d = primary − pooled (> 0: primary гірша)
rows = []
for name, K, s, crn in RUNS:
    out = MC_DIR / name
    if not (out / "forecast_summary.csv").exists():
        continue
    sm = pd.read_csv(out / "forecast_summary.csv").set_index(["model", "scheme"])
    pr = pd.read_csv(out / "forecast_paired_differences.csv")
    q = pr[(pr["model_a"] == "pooled_train") & (pr["model_b"] == "primary_train")].set_index("metric")
    rows.append({"run": name, "K": K, "crn": crn,
                 "MAE_primary": sm.loc[("primary_train", "fixed_origin"), "MAE_race_mean"],
                 "MAE_pooled": sm.loc[("pooled_train", "fixed_origin"), "MAE_race_mean"],
                 "dMAE": -q.loc["mae", "mean_diff_a_minus_b"],
                 "dMAE_ci94_lo": -q.loc["mae", "ci94_hi"], "dMAE_ci94_hi": -q.loc["mae", "ci94_lo"],
                 "dCRPS": -q.loc["crps", "mean_diff_a_minus_b"],
                 "share_races_pooled_better": q.loc["mae", "share_races_a_better"],
                 "dMAE_rolling": sm.loc[("primary_train", "rolling_one_step"), "MAE_race_mean"]
                                 - sm.loc[("pooled_train", "rolling_one_step"), "MAE_race_mean"]})
res_mc = pd.DataFrame(rows).set_index("run")
display(res_mc.round(4))

main_summary = cfg.out_dir / "forecast" / "forecast_summary.csv"
if "orig" in res_mc.index and main_summary.exists():
    o = pd.read_csv(main_summary).set_index(["model", "scheme"])
    ok = all(np.isclose(res_mc.loc["orig", col], o.loc[(m, "fixed_origin"), "MAE_race_mean"])
             for col, m in (("MAE_primary", "primary_train"), ("MAE_pooled", "pooled_train")))
    print("відтворення основного прогону:", "OK" if ok else "НЕ збігається — решту не інтерпретувати")
for label, keys in (("без спільних чисел (orig, seed1, seed2)", ["orig", "seed1", "seed2"]),
                    ("зі спільними числами, K0 (crn1, crn2)", ["crn1", "crn2"]),
                    ("зі спільними числами, 4×K0 (crn1_x4, crn2_x4)", ["crn1_x4", "crn2_x4"])):
    v = res_mc.loc[[k for k in keys if k in res_mc.index], "dMAE"]
    if len(v):
        print(f"dMAE {label}: {v.min():+.4f} … {v.max():+.4f}")
