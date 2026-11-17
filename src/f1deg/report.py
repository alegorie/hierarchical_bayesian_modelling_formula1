"""Ukrainian result summaries, the status of the old conclusions, the experiment registry and the
reviewer-response matrix — all generated from the persisted run.

v1.1: nothing in this module infers that an experiment was run from the MODE.  Every status is
derived from `Pipeline.fit_state()` (checkpoint store + stage registry), so an experiment that was
implemented but never fitted is reported as «не запущено», never as «виконано».
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def fmt_iv(med, lo, hi, nd=3):
    return f"{med:.{nd}f} [{lo:.{nd}f}; {hi:.{nd}f}]"


# --------------------------------------------------------------------------- #
# experiment registry
# --------------------------------------------------------------------------- #
PRIORITY_UK = {"mandatory": "обов'язковий", "recommended": "рекомендований", "optional": "необов'язковий", "": "—"}


def state_label(st: Dict[str, Any]) -> str:
    """One Ukrainian label per experiment, strictly from evidence (see Pipeline.fit_state)."""
    if not st.get("started"):
        return "реалізовано, не запущено"
    if not st.get("completed"):
        return f"запущено, не завершено (ітерація {st.get('iteration')}/{st.get('total_iterations')})"
    if not st.get("diagnosed"):
        return "завершено, діагностику не обчислено"
    if st.get("gates_pass"):
        return "завершено; усі критерії діагностики пройдено"
    if st.get("usable"):
        return "завершено; придатний з застереженням (" + (st.get("failures") or "") + ")"
    return "завершено; критерії діагностики НЕ пройдено (" + (st.get("failures") or "") + ")"


def experiments_status(experiments, states: Dict[str, Dict[str, Any]], hours_per_fit: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    rows = []
    for e in experiments:
        st = states.get(e.name, {})
        rows.append({"experiment": e.name, "tier": e.tier, "priority": e.priority, "пріоритет": PRIORITY_UK.get(e.priority, e.priority), "reviewer_points": e.reviewer_points,
                     "in_queue": st.get("in_queue"), "implemented": True, "started": st.get("started"), "completed": st.get("completed"), "diagnosed": st.get("diagnosed"),
                     "gates_pass": st.get("gates_pass"), "usable": st.get("usable"), "has_ppc": st.get("has_ppc"), "has_loo": st.get("has_loo"),
                     "статус": state_label(st), "worst_rhat": st.get("worst_rhat"), "min_ess": st.get("min_ess"), "divergences": st.get("divergences"),
                     "ebfmi_min": st.get("ebfmi_min"), "sampling_hours": (hours_per_fit or {}).get(e.name), "fit_id": st.get("fit_id", ""),
                     "differs_from_primary": e.differs_from_primary, "question": e.question, "allows": e.allows,
                     "data_variant": str(e.data_variant) if e.data_variant else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# results summary
# --------------------------------------------------------------------------- #
def results_summary_uk(ctx: Dict[str, Any]) -> str:
    """ctx: mode, run_id, counts, global_rate, compound, contrasts, order, scales, residual, diagnostics, sigma_cmp,
    ppc_headline, pit_tails, forecast, forecast_pairs, hierarchy_ablation, identification, sign, loo, interval_prob, state (fit_state of the primary)."""
    L = []
    P = int(ctx["interval_prob"] * 100)
    L.append(f"## Результати (автоматично згенеровано з run {ctx['run_id']}, режим {ctx['mode']})")
    if ctx["mode"] != "FULL":
        L.append(f"> **Увага:** це режим {ctx['mode']} ({'синтетичні дані' if ctx['mode']=='SMOKE' else 'підмножина трас'}). Числа нижче ілюструють формат і НЕ є результатами для статті.")
    c = ctx["counts"]
    L.append(f"Після очищення: {c['laps']:,} кіл, {c['stints']:,} відрізків, {c['races']} гонок, {c['tracks']} трас, {c['drivers']} пілотів, {c['constructors']} конструкторів; {c['innovations']:,} латентних інновацій.")
    d = ctx["diagnostics"]
    g = d["gates"]
    L.append(f"**Діагностика MCMC (primary):** {d['n_chains']} ланцюги × {d['n_draws']} ітерацій після адаптації; найгірший R̂ = {g['worst_rhat']:.4f}, мін. ESS bulk/tail = {g['min_ess_bulk']:.0f}/{g['min_ess_tail']:.0f}, "
             f"макс. MCSE/SD = {g['max_mcse_rel']:.3f}, дивергенцій {d['energy']['divergences']}, E-BFMI min {d['energy']['ebfmi_min']:.3f}. "
             + ("Усі операційні критерії пройдено." if g["pass"] else "Не пройдено: " + "; ".join(g["failures"]) + "."))
    gr = ctx["global_rate"]
    L.append(f"**Загальний рівень швидкості деградації** = {fmt_iv(gr['median'], gr['lo'], gr['hi'])} с/коло ({P}% ETI; функція зв'язку: {gr.get('link', 'log')}).")
    comp = ctx["compound"]
    parts = []
    for _, r in comp.iterrows():
        mult = f"×{fmt_iv(r['multiplier_median'], r['multiplier_lo'], r['multiplier_hi'], 2)}, " if np.isfinite(r["multiplier_median"]) else ""
        parts.append(f"{r['compound']}: {mult}швидкість {fmt_iv(r['rate_median_s_per_lap'], r['rate_lo'], r['rate_hi'])} с/коло, n = {int(r['n_stints'])}")
    L.append("**Компаунди:** " + "; ".join(parts) + ".")
    cc = ctx["contrasts"]
    L.append("**Контрасти (зі спільних апостеріорних реалізацій):** " + "; ".join(f"{r['contrast']}: Δ = {fmt_iv(r['rate_diff_median_s_per_lap'], r['rate_diff_lo'], r['rate_diff_hi'])} с/коло, відношення {fmt_iv(r['ratio_median'], r['ratio_lo'], r['ratio_hi'], 2)}, P(Δ>0) = {r['P(diff>0)']:.3f} ± {r['P_mcse_or_bound']:.3f}" for _, r in cc.iterrows()) + ".")
    od = ctx["order"]
    if len(od):
        gl = od[od["scope"].str.startswith("global")]
        L.append("**Ймовірності порядку (загальний рівень):** " + "; ".join(f"P({r['event']}) = {r['probability']:.4f} (MCSE або межа 3/ESS: {r['mcse_or_rule_of_three_bound']:.4f})" for _, r in gl.iterrows()) + ".")
        tr = od[~od["scope"].str.startswith("global")]
        if len(tr):
            lowp = tr[tr["probability"] < 0.9]
            L.append(f"За трасами (повні швидкості з взаємодією): P(SOFT>MEDIUM>HARD) ≥ 0.9 на {int((tr['probability'] >= 0.9).sum())} з {len(tr)} трас" +
                     ("; нижче 0.9: " + ", ".join(f"{r['scope'].replace('track ','')} ({r['probability']:.2f}; n = {r['n_stints_S/M/H']})" for _, r in lowp.iterrows()) if len(lowp) else "") + ".")
    sc = ctx["scales"]
    hyper = sc[sc["kind"].str.startswith("hyper")]
    L.append("**Масштаби неоднорідності (гіперпараметри σ):** " + "; ".join(f"{r['component']}: {fmt_iv(r['median'], r['lo'], r['hi'])} {r.get('units', '')}" for _, r in hyper.iterrows()) + ".")
    vd = sc[sc["kind"].str.startswith("variance")]
    L.append("**Частки дисперсії лінійного предиктора швидкості між відрізками:** " + "; ".join(f"{r['component']}: {r['median']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}]" for _, r in vd.iterrows()) + ".")
    if ctx.get("sigma_cmp") is not None and len(ctx["sigma_cmp"]):
        r = ctx["sigma_cmp"].iloc[0]
        L.append(f"**Конструктори та пілоти:** {r['comparison']} = {fmt_iv(r['median'], r['lo'], r['hi'])}, P(σ_G > σ_D) = {r['P(>0)']:.2f}.")
    rs = ctx["residual"].set_index("site") if len(ctx["residual"]) else None
    if rs is not None:
        txt = []
        for s, lab in (("sigma_eps_core", "σ_core"), ("lambda_skew", "λ"), ("nu_df", "η"), ("sigma_eta", "σ_η"), ("sigma_stint_base", "σ_β"), ("baseline_mean", "β₀")):
            if s in rs.index:
                r = rs.loc[s]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                txt.append(f"{lab} = {fmt_iv(r['median'], r['lo'], r['hi'], 4)}")
        L.append("**Похибки спостережень, процесний шум, базові рівні:** " + "; ".join(txt) + ".")
    if ctx.get("identification"):
        idn = ctx["identification"]
        if "corr_mu_log_nu__mu_log_gamma" in idn:
            L.append(f"**Розділення ν і γ:** corr(μ_ν, μ_γ) = {idn['corr_mu_log_nu__mu_log_gamma']:.2f}; SD(μ_γ) апостеріорне/апріорне = {idn['mu_log_gamma_sd_ratio_post_over_prior']:.2f}; "
                     f"середня за трасами кореляція трасової швидкості та γ_q = {idn['corr_track_rate_gamma_mean']:.2f}.")
    sg = ctx.get("sign")
    if sg is not None and len(sg):
        free = sg[~sg["by_construction"]]
        if len(free) <= 2:
            L.append("**Знак швидкості:** у цій специфікації (log-зв'язок) ν_s > 0 за побудовою на всіх рівнях; це припущення моделі, а не емпіричний висновок. Емпірична перевірка — експеримент `signed_full` (тотожна функція зв'язку).")
        else:
            L.append("**Знак швидкості (специфікація допускає ν ≤ 0):** " + "; ".join(f"{r['level']}: {r['quantity']} = {r['value']:.3f}" for _, r in free.iterrows()) + ".")
    ph = ctx.get("ppc_headline") or {}
    if ph:
        L.append("**Умовна апостеріорна предиктивна перевірка (репліки усічено тим самим правилом, що й дані):** " + "; ".join(f"{k}: спостережено {v['observed']:.3f}, репліки {v['rep_mean']:.3f} (p = {v['p']:.2f})" for k, v in ph.items()) + ".")
    pt = ctx.get("pit_tails") or {}
    if pt:
        L.append(f"**PIT:** частка спостережень у верхніх 5 % / 1 % умовного предиктивного розподілу: {pt.get('pit_gt_0.95', float('nan')):.3f} / {pt.get('pit_gt_0.99', float('nan')):.3f} (номінал 0.05 / 0.01); у нижніх: {pt.get('pit_lt_0.05', float('nan')):.3f} / {pt.get('pit_lt_0.01', float('nan')):.3f}.")
    loo = ctx.get("loo")
    if loo:
        L.append(f"**PSIS-LOO (умовно на латентні стани):** elpd = {loo['elpd_loo']:.1f} (SE {loo['se_elpd_loo']:.1f}), p_loo = {loo['p_loo']:.1f}; частка кіл з k̂ > 0.7: {loo['share_k_gt_0.7']:.3f}"
                 + (" — оцінка НЕНАДІЙНА, використовувати лише разом із прогнозною перевіркою." if not loo.get("reliable") else "."))
    fc = ctx.get("forecast")
    if fc is not None and len(fc):
        f = fc[fc["scheme"] == "fixed_origin"]
        L.append("**Прогноз (фіксований початок; остання чверть відрізків тестового сезону лише з префікса):** " + "; ".join(
            f"{r['model']}: MAE {r['MAE']:.3f}, RMSE {r['RMSE_pooled']:.3f}, CRPS {r['CRPS']:.3f}" + (f", LPD {r['LPD']:.3f}" if "LPD" in r and np.isfinite(r["LPD"]) else "") + f", покриття94 {r['coverage94']:.3f}" for _, r in f.iterrows()) + ".")
    fp = ctx.get("forecast_pairs")
    if fp is not None and len(fp):
        L.append("**Парні різниці за гонками (a − b; інтервал за t-розподілом):** " + "; ".join(
            f"{r['model_a']} − {r['model_b']} [{r['metric']}]: {r['mean_diff_a_minus_b']:+.4f} (SE {r['se_race']:.4f}; [{r['ci94_lo']:+.4f}; {r['ci94_hi']:+.4f}])" for _, r in fp.iterrows()) + ".")
    ha = ctx.get("hierarchy_ablation")
    if ha is not None and len(ha):
        # v1.2: pooled_train (no group effects) -> unpooled_train (effects, fixed scales) -> primary_train (learned scales)
        main = ha[(ha["scheme"].str.contains("основна схема")) & (ha["stints"] == "усі відрізки")]
        lab = {"mae": "MAE", "crps": "CRPS", "lpd": "LPD"}
        parts = []
        for cmp_, g in main.groupby("comparison", sort=False):
            parts.append(cmp_ + ": " + "; ".join(f"{lab.get(r['metric'], r['metric'])} {r['mean_diff_a_minus_b']:+.4f} [{r['ci94_lo']:+.4f}; {r['ci94_hi']:+.4f}] — {r['reading']}" for _, r in g.iterrows()))
        if parts:
            L.append("**Чи допомагає прогнозу адаптивне ієрархічне об'єднання (основна схема, усі відрізки; MAE і CRPS — втрати, LPD — більше краще):** " + ". ".join(parts)
                     + ". Інтервал, що містить нуль, означає лише, що різницю не підтверджено на цих гонках, а не що моделі рівноцінні. Інші схеми — forecast/forecast_hierarchy_ablation.csv.")
    return "\n\n".join(L)


# --------------------------------------------------------------------------- #
# status of the conclusions of the first version of the paper
# --------------------------------------------------------------------------- #
OLD_CONCLUSIONS = [
    ("Модель «верифіковано», вона «успішно ізолює чистий додатний латентний дрейф»", "recovery+identification"),
    ("«Повна збіжність» ланцюгів (R̂=1.09, ESS=43 для σ_C)", "diagnostics"),
    ("Геометрична середня швидкість ≈ 0.067 с/коло [0.062; 0.071]", "global_rate"),
    ("Множники Soft/Medium/Hard ≈ 1.31/0.94/0.82; Soft у ~1.6 раза швидше за Hard", "compound"),
    ("Порядок Soft > Medium > Hard з «дуже високою ймовірністю»", "order"),
    ("«Статистична значущість та незворотність» деградації", "signed"),
    ("Найбільше джерело варіації — траса (σ_T ≈ 0.444)", "scales"),
    ("«Суттєва значущість» взаємодії траса–компаунд", "interaction"),
    ("Конструктори «значно стабільніші» за пілотів (σ_G ≈ 0.097 vs σ_D ≈ 0.042)", "sigma_cmp"),
    ("Переоцінена правостороння скошеність — «очікуваний» ефект", "ppc"),
    ("γ_q найбільший там, де покриття «накатується» (Монако) — інтерпретація як еволюції траси", "identification"),
    ("Придатність як апріорного розподілу для онлайн-фільтрації та стратегій", "forecast"),
]


def old_conclusions_status(ctx: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    usable = ctx["mode"] == "FULL" and bool((ctx.get("state") or {}).get("completed"))
    for text, key in OLD_CONCLUSIONS:
        status, note = "не перевірено", "потрібен завершений FULL-fit `primary` на повних даних"
        if usable:
            if key == "diagnostics":
                g = ctx["diagnostics"]["gates"]
                status = "змінено"
                note = (f"найгірший R̂ {g['worst_rhat']:.4f}, мін. ESS {min(g['min_ess_bulk'], g['min_ess_tail']):.0f}; " +
                        ("усі критерії пройдено" if g["pass"] else "не пройдено: " + "; ".join(g["failures"])) + "; формулювання «повна збіжність» прибрано")
            elif key == "global_rate":
                gr = ctx["global_rate"]; status = "підтверджено" if 0.062 <= gr["median"] <= 0.071 else "змінено"
                note = f"нове значення {fmt_iv(gr['median'], gr['lo'], gr['hi'])} с/коло (інша специфікація прайорів і бібліотека; порівнянність обмежена)"
            elif key == "compound":
                comp = ctx["compound"].set_index("compound")
                try:
                    r = comp.loc["SOFT", "rate_median_s_per_lap"] / comp.loc["HARD", "rate_median_s_per_lap"]
                    status = "підтверджено" if 1.4 <= r <= 1.8 else "змінено"; note = f"відношення медіан швидкостей SOFT/HARD = {r:.2f}; ефекти компаундів — із фіксованою апріорною шкалою 0.5 (сума нуль)"
                except KeyError:
                    pass
            elif key == "order":
                od = ctx["order"]; gl = od[od["event"] == "SOFT > MEDIUM > HARD"] if len(od) else od
                if len(gl):
                    p = float(gl["probability"].iloc[0]); status = "підтверджено (на загальному рівні)" if p >= 0.95 else "послаблено"
                    tr = od[~od["scope"].str.startswith("global")]
                    note = f"P = {p:.4f} (MCSE/межа {float(gl['mcse_or_rule_of_three_bound'].iloc[0]):.4f})" + (f"; за трасами P ≥ 0.9 лише на {int((tr['probability'] >= 0.9).sum())} з {len(tr)}" if len(tr) else "")
            elif key == "signed":
                st = ctx.get("states", {}).get("signed_full", {})
                status = "змінено"
                note = "додатність ν — припущення log-параметризації; формулювання прибрано. " + ("Перевірку специфікацією з ν ≤ 0 виконано (signed_full)." if st.get("completed") else "Специфікацію з ν ≤ 0 (signed_full) реалізовано, але НЕ запущено.")
            elif key == "scales":
                sc = ctx["scales"]; h = sc[(sc["component"] == "траса") & sc["kind"].str.startswith("hyper")]
                if len(h):
                    top = sc[sc["kind"].str.startswith("variance")].sort_values("median", ascending=False)
                    status = "підтверджено" if len(top) and top["component"].iloc[0] == "траса" else "змінено"
                    note = f"σ_T = {fmt_iv(h['median'].iloc[0], h['lo'].iloc[0], h['hi'].iloc[0])}" + (f"; частка дисперсії {top['median'].iloc[0]:.2f} [{top['lo'].iloc[0]:.2f}; {top['hi'].iloc[0]:.2f}]" if len(top) else "")
            elif key == "interaction":
                sc = ctx["scales"]; h = sc[(sc["component"] == "траса×компаунд") & sc["kind"].str.startswith("hyper")]
                if len(h):
                    status = "послаблено"; note = f"σ_I = {fmt_iv(h['median'].iloc[0], h['lo'].iloc[0], h['hi'].iloc[0])}; «значущість» замінено величиною ефекту з невизначеністю та кількістю відрізків у комірках"
            elif key == "sigma_cmp" and ctx.get("sigma_cmp") is not None and len(ctx["sigma_cmp"]):
                r = ctx["sigma_cmp"].iloc[0]; status = "не підтверджено" if r["P(>0)"] < 0.975 else "підтверджено"
                note = f"σ_G − σ_D = {fmt_iv(r['median'], r['lo'], r['hi'])}, P(σ_G > σ_D) = {r['P(>0)']:.2f}"
            elif key == "ppc":
                status = "змінено"; note = "перевірку переформульовано: усічені репліки, квантилі, частки перевищень, PIT; вибіркова скошеність ненадійна при η → 2"
            elif key == "identification" and ctx.get("identification") and "corr_mu_log_nu__mu_log_gamma" in ctx["identification"]:
                status = "послаблено"; note = f"corr(μ_ν, μ_γ) = {ctx['identification']['corr_mu_log_nu__mu_log_gamma']:.2f}; фізична інтерпретація γ_q — гіпотеза"
            elif key == "forecast":
                status = "послаблено"; note = "виконано лише ретроспективний прогноз із префікса відрізка; онлайн-фільтрація — напрям подальшої роботи"
            elif key == "recovery+identification":
                status = "змінено"; note = "«верифіковано» замінено переліком виконаних перевірок (відновлення на синтетичних даних, предиктивні перевірки, прогноз) і явними умовами ідентифіковності"
        rows.append({"старий висновок": text, "статус": status, "коментар": note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# reviewer-response matrix
# --------------------------------------------------------------------------- #
# code   A1..A9 = «Обов'язкові виправлення» 1–9 рецензента A;  A10..A12 = «Побажання» 1–3;
#        A13..A16 = зауваження з критеріїв, яких немає у підсумковому переліку;  B1..B5 = рецензент B;  E1..E3 = редакція.
# needs  experiments without which the remark is not answered;  wants = recommended additional experiments.
#        Invariants (tests/rev2): union(needs) == mandatory experiments; every `wants` entry is mandatory or recommended;
#        every recommended experiment is wanted by at least one remark.
REVIEWER_POINTS: List[Dict[str, Any]] = [
    {"code": "A1", "remark": "Збіжність MCMC: R̂ = 1,09 для σ_C, ESS = 43, E-BFMI 0,30–0,32; навести R̂ та ESS (bulk/tail) для всіх гіперпараметрів, кількість ланцюгів, довжину адаптації й вибірки; прибрати «повна збіжність».",
     "action": "Модель перенесено на NumPyro/JAX; нецентрована параметризація всіх ієрархічних блоків, щільний блок матриці мас для гіперпараметрів, 4 ланцюги × (2000 + 2000). Ефекти компаунда — з фіксованою апріорною шкалою (три рівні не дають оцінити σ_C). Таблиця R̂/ESS bulk/tail/MCSE для кожного гіперпараметра генерується автоматично; операційні критерії перевіряються на неокруглених значеннях. Центрована параметризація латентного стану реалізована як необов'язковий діагностичний експеримент.",
     "code_ref": "sampler.py; diagnostics.py (hyper_table, evaluate_gates); model.py (latent_param)", "paper_ref": "«Оцінювання параметрів і перевірка моделі» (Обчислення); «Результати та обговорення» (Збіжність); табл. 3, 7",
     "artifacts": "tables/mcmc_hyper_primary.csv; tables/diagnostics_gates.csv; paper/tab3_mcmc.md", "needs": ["primary"], "wants": [],
     "note": "Низький E-BFMI основної моделі подано в статті як застереження; специфікація з похибками Стьюдента, яка виконує всі критерії, дає близькі оцінки. Центрована параметризація (primary_centred) — необов'язковий діагностичний експеримент: у пілотній перевірці вона змішувалася гірше за нецентровану"},
    {"code": "A2", "remark": "Додатність ν_s закладена log-параметризацією; «статистична значущість та незворотність» не є висновком із даних. Змінити формулювання або показати чутливість до специфікації з ν ≤ 0 і порівняти за LOO/WAIC.",
     "action": "Формулювання прибрано; у тексті прямо сказано, що ν_s > 0 — припущення. Додано специфікацію з тотожною функцією зв'язку (усі ефекти адитивні в с/коло, знак вільний на кожному рівні), таблицю знаків (частка апостеріорної маси ≤ 0 за рівнями), PSIS-LOO/WAIC та прогнозну логарифмічну оцінку для порівняння зв'язків; сценарії відновлення identity_rates та identity_zero.",
     "code_ref": "model.py (rate_link='identity'); link.py; results.sign_table; loo.py", "paper_ref": "«Математична модель» (Альтернативні специфікації: формула (10), табл. 2); «Швидкість деградації та компаунди»; «Обмеження»",
     "artifacts": "tables/sign_*.csv; tables/loo_comparison.csv; tables/synthetic_recovery_summary*.csv", "needs": [], "wants": ["signed_full"],
     "note": "Мінімальна відповідь (зміна формулювань) виконана в тексті; емпірична частина потребує запуску signed_full."},
    {"code": "A3", "remark": "Базові моделі та валідація: МНК із паливною корекцією, модель без ієрархії/взаємодії, нормальні похибки, модель [1]; відновлення параметрів на синтетичних даних (зокрема розділення ν і γ); прогноз на відкладених гонках; замінити «верифіковано».",
     "action": "МНК із зовнішньою паливною поправкою; моделі без взаємодії, без ієрархії, з нормальними похибками та похибками Стьюдента, з фіксованим γ; адаптація моделі Cappello–Hoegh (швидкість лише за компаундом); модель із тими самими груповими ефектами, але з фіксованими масштабами (unpooled: σ_T, σ_D, σ_G, σ_I не оцінюються) — щоб відокремити внесок самого адаптивного об'єднання від внеску групових ефектів. Відновлення параметрів незалежним генератором, зокрема сценарії з відомим γ. Прогноз: навчання на 2022–2023, перевірка на 2024 (останні кола відрізків лише з префікса), MAE/RMSE/CRPS/LPD/покриття, парні різниці за гонками. «Верифіковано» замінено переліком перевірок.",
     "code_ref": "baseline.py; config.default_experiments; model.py (group_scale='fixed'); recovery.py; synthetic.py; forecast.py (hierarchy_ablation); scripts/run_recovery_study.py", "paper_ref": "«Ідентифіковність швидкості деградації»; табл. 2; «Порівняння специфікацій» (табл. 7); «Прогноз» (табл. 8, рис. 7); «Відновлення параметрів» (табл. 9, рис. 8)",
     "artifacts": "tables/model_comparison.csv; forecast/forecast_summary*.csv; forecast/forecast_paired_differences*.csv; forecast/forecast_hierarchy_ablation.csv; tables/synthetic_recovery_summary*.csv",
     "needs": ["primary", "noint", "gauss", "gamma_fixed", "primary_train", "gauss_train", "pooled_train", "ch_adapted_train"], "wants": ["noint_train", "unpooled_train", "unpooled"],
     "note": "Два результати перевірок обмежують висновки й прямо названі в статті: (1) групові ефекти траси, конструктора й пілота не поліпшили прогноз сезону 2024 року порівняно "
             "з моделлю без ієрархії (ця модель не має групових ефектів узагалі; чи допомагає саме навчання масштабів, покаже порівняння primary_train з unpooled_train); (2) на синтетичних даних відносні ефекти відновлюються з покриттям, близьким до номінального, але загальний рівень швидкості на малих "
             "вибірках занижується (зміщення зменшується зі зростанням вибірки; для вибірки реального розміру перевірку не виконано). Числа — у табл. 8, 9 і в "
             "tables/synthetic_recovery_summary*.csv (за відсутності власного дослідження — results_sandbox/)."},
    {"code": "A4", "remark": "Повнота результатів: усі прайори в табл. 1, пояснити HN(1; 0,5²) для λ; додати до табл. 2 σ_I, σ_β, σ_η, σ, λ, η, γ_q; числові P(Soft > Medium > Hard) та підтвердження «значущості взаємодії».",
     "action": "Табл. 1 генерується з коду (єдине джерело з моделлю) і містить усі прайори; λ ~ TruncN(0,3; 0,25²) на (−0,95; 0,95) (HN(1; 0,5²) у першій версії — помилка опису). Таблиця результатів містить усі названі параметри; γ_q за трасами — окрема таблиця. Ймовірності порядку з MCSE на загальному рівні та за трасами.",
     "code_ref": "model.prior_table; results.compound_tables, residual_table, track_table", "paper_ref": "табл. 1, 3–5; «Швидкість деградації та компаунди»; «Взаємодія „траса — компаунд“»",
     "artifacts": "tables/priors_primary.csv; tables/order_probabilities_primary.csv; tables/residual_primary.csv; tables/tracks_primary.csv", "needs": ["primary"], "wants": []},
    {"code": "A5", "remark": "Взаємодія «траса–компаунд» (рис. 6): невизначеність і кількість відрізків у комірці, кольорова шкала; пояснити немонотонні комірки (Imola, Montréal, Lusail); обґрунтувати ієрархічний прайор σ_C з трьох рівнів або перейти до фіксованих ефектів.",
     "action": "Теплова карта з n та шириною інтервалу в кожній комірці, шкала без насичення; таблиця повних швидкостей за комірками й P(порядку) за трасами; комірки з n < 5 позначено як слабко інформовані. Немонотонність подано як відкрите питання із зазначенням змішування компаунда з фазою гонки. Ефекти компаунда — фіксовані (сума нуль, шкала 0,5); ієрархічний σ_C лишено як перевірку чутливості.",
     "code_ref": "results.interaction_table; export.fig_interaction; config (comp_hier)", "paper_ref": "«Взаємодія „траса — компаунд“» (рис. 5); табл. 5",
     "artifacts": "tables/interaction_primary.csv; figures/fig06_interaction_primary.*", "needs": ["primary", "noint"], "wants": ["noint_train"]},
    {"code": "A6", "remark": "Конструктори проти пілотів: інтервали σ_G і σ_D перекриваються, «значно стабільніші» не підтверджується; фізичні пояснення — як гіпотези; додати конструктор і взаємодію на рис. 5.",
     "action": "Порівняння подано апостеріорною різницею σ_G − σ_D та P(σ_G > σ_D); твердження прибрано. Фізичні пояснення перенесено в гіпотези. Рисунок масштабів містить усі компоненти та розклад дисперсії.",
     "code_ref": "results.sigma_comparison, scales_table; export.fig_scales", "paper_ref": "«Джерела неоднорідності»; «Конструктори та пілоти»; табл. 5, рис. 4",
     "artifacts": "tables/sigma_comparison_primary.csv; tables/scales_primary.csv; figures/fig05_scales_primary.*", "needs": ["primary"], "wants": []},
    {"code": "A7", "remark": "Апостеріорні перевірки: навести й обговорити p = 0,94 і скошеність реплік до 200 (надто важкі хвости); чутливість до порога 1,30 та мінімуму п'яти кіл.",
     "action": "Предиктивні перевірки на квантилях, IQR/MAD, частках перевищень і PIT з репліками, усіченими як дані; вибіркова скошеність подається з поясненням її ненадійності при η → 2. Оцінка η на межі носія названа обмеженням; порівняння з похибками Стьюдента й нормальними. Чутливість до порога та мінімуму кіл: на рівні даних і МНК — виконано точно; пороги 1,20 і 1,40 змінюють лише 36 і 19 кіл із 57 143, тому для повторного оцінювання повної моделі обрано пороги, що справді змінюють вибірку (1,10 — мінус 467 кіл; 1,07 — мінус 3 560), і мінімум 8 та 10 кіл.",
     "code_ref": "ppc.py; config (clean_110/107, minlaps_8/10; необов'язкові clean_120/140, skewt_df5); pipeline.stage_data_sensitivity", "paper_ref": "«Предиктивні перевірки»; «Похибки та предиктивні перевірки» (табл. 6, рис. 6); «Чутливість до правил відбору»; «Обмеження»",
     "artifacts": "ppc/ppc_conditional_*.csv; tables/sensitivity.csv; tables/data_threshold_sensitivity.csv", "needs": ["primary", "gauss", "clean_110", "minlaps_8"], "wants": ["studentt", "clean_107", "minlaps_10"]},
    {"code": "A8", "remark": "Література: ≥ 50 % джерел за останні п'ять років; праці зі стратегій і деградації шин; критичне порівняння з [1]; єдине оформлення; перевірка бібліографічних даних.",
     "action": "Список переглянуто (IEEE), додано праці з моделювання гонок, стратегій, зносу шин і беєсівського робочого процесу; окремий підрозділ порівняння з моделлю Cappello–Hoegh. Бібліографічні дані нових джерел перевірено за сторінками видавців; перелік неперевірених полів — у docs/REFERENCES_CHECK_uk.md.",
     "code_ref": "—", "paper_ref": "«Аналіз останніх досліджень і публікацій»; «Порівняння з результатами інших досліджень»; Список літератури", "artifacts": "docs/REFERENCES_CHECK_uk.md", "needs": [], "wants": [], "author_action": "остаточно звірити DOI/сторінки за переліком"},
    {"code": "A9", "remark": "Відтворюваність: README, ліцензія, версії бібліотек, seed; версія з постійним ідентифікатором (DOI).",
     "action": "README.md, requirements.txt, LICENSE (чернетка MIT — рішення авторів), CITATION.cff; seed і версії фіксуються в маніфесті кожного fit; відбитки даних і специфікацій.",
     "code_ref": "README.md; requirements.txt; LICENSE; CITATION.cff; sampler.py (manifest)", "paper_ref": "«Інформація щодо наборів даних»", "artifacts": "fits/*/manifest.json", "needs": [], "wants": [],
     "author_action": "опублікувати реліз і отримати DOI (Zenodo); вписати DOI у статтю та CITATION.cff"},
    {"code": "A10", "remark": "Побажання: скоротити анотацію; уніфікувати шкали (с/коло та лог с/коло) і написання «баєсівський/беєсівський»; розширити ключові слова.",
     "action": "Анотацію скорочено; у таблицях окремо подано швидкості (с/коло), масштаби (лог с/коло) і безрозмірні множники; у тексті одне написання; ключові слова оновлено («часткове об'єднання», «розподіл Гансена», «MCMC»).",
     "code_ref": "results.py (колонка units)", "paper_ref": "анотація, ключові слова, табл. 4–5", "artifacts": "—", "needs": [], "wants": [], "author_action": "підтвердити обраний варіант написання"},
    {"code": "A11", "remark": "Побажання: визначення «відносного віку шини», вилучені гонки (68 → 66), перейменовані команди; обговорити нелінійність зносу та нормування на довжину кола.",
     "action": "Визначення віку (TyreLife, перше збережене коло, відносний вік, реальні прирости Δa), перелік вилучених гонок із причиною, таблиця зіставлення команд, частка вживаних комплектів. Лінійність зносу — явне припущення з перевіркою залишків за віком; трасові швидкості додатково подано у % часу кола.",
     "code_ref": "data.py; pipeline.stage_data; results.track_table", "paper_ref": "«Дані та попередня обробка»; «Джерела неоднорідності»; «Обмеження»",
     "artifacts": "tables/data_exclusions.csv; tables/data_team_map.csv; tables/data_definitions.csv; tables/tracks_primary.csv; ppc/ppc_conditional_residuals_primary.csv", "needs": ["primary"], "wants": []},
    {"code": "A12", "remark": "Побажання: рисунки — одиниці на осях (с/коло), легенди, шрифти, усічений підпис.",
     "action": "Усі рисунки будуються автоматично з таблиць запуску: одиниці «с/коло», повні легенди, єдиний шрифт, українські підписи, кольори й маркери компаундів.",
     "code_ref": "export.py", "paper_ref": "рис. 1–8", "artifacts": "figures/*.png|pdf; figures/captions_uk.json", "needs": ["primary"], "wants": []},
    {"code": "A13", "remark": "Кількість ланцюгів, довжина адаптації та вибірки (8000 ітерацій — на ланцюг чи сумарно?), версії бібліотек.",
     "action": "У статті: 4 ланцюги × (2000 адаптації + 2000 збережених) = 8000 збережених реалізацій сумарно; NumPyro, JAX, Python — версії з маніфесту.",
     "code_ref": "sampler.py (manifest); config.yaml", "paper_ref": "«Оцінювання параметрів і перевірка моделі» (Обчислення)", "artifacts": "fits/*/manifest.json; config_used.yaml", "needs": ["primary"], "wants": []},
    {"code": "A14", "remark": "Практична цінність для стратегії піт-стопів заявлена, але не проілюстрована.",
     "action": "Додано прогноз останніх кіл відрізка за його початком: сукупні оцінки (MAE, RMSE, CRPS, покриття, парні різниці за гонками) — у статті, приклади відрізків — у figures/; підрозділ про шляхи застосування. Чесно зазначено, що групові ефекти не дали прогнозного виграшу над моделлю без ієрархії на сезоні 2024 року.",
     "code_ref": "forecast.py; export.fig_examples", "paper_ref": "«Прогноз» (табл. 8, рис. 7); «Шляхи практичного застосування»", "artifacts": "forecast/*.csv; figures/fig08_examples_forecasting_forecast.*", "needs": ["primary_train", "pooled_train"], "wants": ["unpooled_train"]},
    {"code": "A15", "remark": "Мова: стилістичні неточності, непослідовна українізація термінів (harvesting, lift-and-coast, андеркат).",
     "action": "Текст переписано. Жаргонні англомовні терміни (harvesting, lift-and-coast, андеркат) у новій редакції не вживаються; stint — «відрізок», compound — «компаунд», partial pooling — «часткове об'єднання»; скорочення розшифровано при першій згадці.", "code_ref": "—", "paper_ref": "увесь текст", "artifacts": "—", "needs": [], "wants": []},
    {"code": "A16", "remark": "Підтвердити бібліографічні дані всіх джерел (заявлено допоміжну систематизацію літератури за допомогою ШІ).",
     "action": "Див. A8.", "code_ref": "—", "paper_ref": "Список літератури; заява про використання ШІ", "artifacts": "docs/REFERENCES_CHECK_uk.md", "needs": [], "wants": [], "author_action": "підтвердити в листі-відповіді"},
    {"code": "B1", "remark": "Немає порівняння експериментальних результатів із результатами аналогічних досліджень.",
     "action": "Підрозділ порівняння з моделлю Cappello–Hoegh (швидкості за компаундами, специфікація похибок, спосіб перевірки) та з працями з моделювання гонок; адаптація моделі Cappello–Hoegh у спільній схемі прогнозу.",
     "code_ref": "config (ch_adapted, ch_adapted_train)", "paper_ref": "«Порівняння з результатами інших досліджень»", "artifacts": "forecast/forecast_summary.csv", "needs": [], "wants": ["ch_adapted_train"],
     "note": "Текстове порівняння з опублікованими числами Cappello–Hoegh виконано; числове порівняння на спільних даних потребує ch_adapted_train."},
    {"code": "B2", "remark": "Не визначено шляхів майбутнього практичного впровадження.", "action": "Додано підрозділ «Практичне застосування» та відповідний абзац у висновках.",
     "code_ref": "—", "paper_ref": "«Шляхи практичного застосування»; «Висновки»", "artifacts": "—", "needs": [], "wants": []},
    {"code": "B3", "remark": "Виокремити підрозділи «Вступ», «Огляд літератури / Аналіз досліджень», «Мета дослідження».", "action": "Структуру змінено.", "code_ref": "—", "paper_ref": "«Вступ»; «Аналіз останніх досліджень і публікацій»; «Мета дослідження»", "artifacts": "—", "needs": [], "wants": []},
    {"code": "B4", "remark": "Менше 50 % джерел за останні п'ять років.", "action": "Див. A8.", "code_ref": "—", "paper_ref": "Список літератури", "artifacts": "docs/REFERENCES_CHECK_uk.md", "needs": [], "wants": []},
    {"code": "B5", "remark": "Бібліографічні описи джерел 2 і 3 оформлено за іншим стандартом.", "action": "Увесь список — за IEEE.", "code_ref": "—", "paper_ref": "Список літератури", "artifacts": "—", "needs": [], "wants": []},
    {"code": "E1", "remark": "«Список літератури» та «References (transliterated)» — за стилем IEEE.", "action": "Виконано.", "code_ref": "—", "paper_ref": "Список літератури; References (transliterated)", "artifacts": "—", "needs": [], "wants": []},
    {"code": "E2", "remark": "Не менше 50 % джерел — за останні п'ять років.", "action": "Див. A8 (частка наведена в docs/REFERENCES_CHECK_uk.md).", "code_ref": "—", "paper_ref": "Список літератури", "artifacts": "docs/REFERENCES_CHECK_uk.md", "needs": [], "wants": []},
    {"code": "E3", "remark": "Додати розділ «References (transliterated)».", "action": "Додано.", "code_ref": "—", "paper_ref": "References (transliterated)", "artifacts": "—", "needs": [], "wants": []},
]


def point_status(pt: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> str:
    done = lambda n: bool(states.get(n, {}).get("completed"))
    need, want = pt.get("needs", []), pt.get("wants", [])
    miss, wmiss = [n for n in need if not done(n)], [n for n in want if not done(n)]
    parts = []
    if not need and not want:
        parts.append("виконано в коді/тексті")
    else:
        if need and not miss:
            parts.append("обов'язкові експерименти виконано: " + ", ".join(need))
        elif need:
            ok = [n for n in need if done(n)]
            parts.append("ЧАСТКОВО: виконано " + (", ".join(ok) if ok else "—") + "; НЕ запущено: " + ", ".join(miss))
        elif want and not wmiss:
            parts.append("виконано разом із рекомендованими експериментами")
        else:
            parts.append("текстову/кодову частину виконано")
        if wmiss:
            parts.append("рекомендовані не запущено: " + ", ".join(wmiss))
        fam = {"rhat": "R̂", "ess": "ESS", "mcse": "MCSE", "divergences": "дивергенції", "ebfmi": "E-BFMI"}
        flagged = [f"{n} ({', '.join(fam.get(k, k) for k in states[n].get('families_failed') or []) or states[n].get('failures')})"
                   for n in need + want if done(n) and states[n].get("diagnosed") and not states[n].get("gates_pass")]
        if flagged:
            parts.append("не всі критерії діагностики пройдено: " + "; ".join(flagged))
    if pt.get("note"):
        parts.append(pt["note"])
    if pt.get("author_action"):
        parts.append("дія авторів: " + pt["author_action"])
    return ". ".join(parts)


def reviewer_response_matrix(states: Dict[str, Dict[str, Any]], out_path: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for pt in REVIEWER_POINTS:
        rows.append({"код": pt["code"], "зауваження": pt["remark"], "що зроблено": pt["action"], "код/файли": pt["code_ref"], "розділ статті": pt["paper_ref"],
                     "артефакти": pt["artifacts"], "потрібні експерименти": ", ".join(pt.get("needs", [])), "рекомендовані експерименти": ", ".join(pt.get("wants", [])),
                     "статус": point_status(pt, states)})
    df = pd.DataFrame(rows)
    if out_path is not None:
        df.to_csv(out_path, index=False)
        try:
            Path(out_path).with_suffix(".md").write_text("**Матриця відповіді рецензентам** (A1–A9 — обов'язкові виправлення рецензента A; A10–A12 — побажання; A13–A16 — зауваження з критеріїв; B — рецензент B; E — редакція). У формулюваннях зауважень «[1]» — нумерація першої версії рукопису (модель Cappello–Hoegh).\n\n"
                                                         + df.to_markdown(index=False) + "\n", encoding="utf-8")
        except Exception:
            pass
    return df


def editorial_checklist_md() -> str:
    return """# Дії авторів, які не можна виконати автоматично

- [ ] Підтвердити варіант написання «беєсівський» (ужито в усьому тексті, як у назві поданої статті) або замінити на «баєсівський» скрізь.
- [ ] Обрати ліцензію коду (у репозиторії — чернетка MIT) і вказати умови використання даних FastF1.
- [ ] Опублікувати реліз репозиторію, отримати DOI (Zenodo), вписати його в статтю («Інформація щодо наборів даних») та в CITATION.cff.
- [ ] Звірити поля бібліографічних описів, позначені в docs/REFERENCES_CHECK_uk.md як неперевірені.
- [ ] Перевірити обсяг статті за вимогами журналу (текст після доопрацювання довший).
- [ ] Після запуску обов'язкових експериментів (docs/RUN_PLAN_uk.md) виконати `python scripts/run_pipeline.py --mode REPORT_ONLY` і замінити в статті позначені жовтим місця числами з `outputs_local/full/paper/`.
- [ ] Оновити заяву про використання ШІ відповідно до фактично використаних інструментів.
- [ ] Після `REPORT_ONLY` замінити рисунки статті командою `python scripts/update_docx_figures.py … --drop-notes` (рис. 3, 5, 6 зараз узято з попереднього запуску).
- [ ] У шаблоні журналу колонтитули парних і непарних сторінок містять різні номери випуску («№ 1. 2026» і «№ 2. 2026») — уточнити в редакції; у файлі статті їх не змінено.
"""
