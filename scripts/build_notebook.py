#!/usr/bin/env python
"""Builds f1_tyre_degradation_revision.ipynb from source (kept here so the notebook
itself stays reproducible; run `python scripts/build_notebook.py`).

v1.1 (після рецензії): комірки відповідають оновленому notebook — черга оцінювань через `cfg.fits_override`,
реєстр експериментів із фактичним станом, `pl.postprocess()` (LOO, таблиці статті), дослідження відновлення.

    python scripts/build_notebook.py                 # за замовчуванням f1_tyre_degradation_revision.ipynb
    python scripts/build_notebook.py out.ipynb       # інший файл
    python scripts/build_notebook.py --force         # дозволити заміну notebook, що містить збережені виходи

Захист результатів: якщо цільовий файл уже існує й містить виходи комірок (наприклад, запуск FULL), скрипт
його НЕ замінює, а пише поруч файл із суфіксом `.clean.ipynb` (потрібен явний `--force`).
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Ієрархічна беєсівська модель простору станів для деградації шин F1 (2022–2024) — відтворюваний notebook редакції (v1.1)

**Призначення.** Обчислювальна основа доопрацювання статті після рецензування: реалізація основної моделі та її альтернативних специфікацій, діагностика MCMC, апріорні й апостеріорні предиктивні перевірки, PSIS-LOO, дослідження відновлення параметрів на синтетичних даних, реєстр експериментів, часово коректна прогнозна перевірка, автоматичне формування рисунків, таблиць статті та матеріалів для відповіді рецензентам.

**Як користуватись.**
1. Покладіть `f1_cache_combined_2022_2024.csv` у `data/` (для `SMOKE` не потрібно — синтетичні дані).
2. Виставте режим у комірці конфігурації нижче (`MODE`): `SMOKE` (хвилини, синтетика, перевірка всього ланцюжка), `PILOT` (3 траси, реальні дані, оцінка часу), `FULL` (усі дані), `REPORT_ONLY` (без MCMC: діагностика, предиктивні перевірки, LOO, прогноз, таблиці й рисунки для всіх уже завершених оцінювань).
3. `Kernel → Restart & Run All`. **Після вимкнення світла: просто запустіть `Run All` знову** (або `python scripts/run_pipeline.py --mode FULL`). Усі завершені етапи пропускаються, незавершене оцінювання продовжується з останнього зафіксованого checkpoint (повний стан sampler, без повторної адаптації; втрата — щонайбільше `checkpoint_every` переходів).

**Чесність результатів.** Жодне число в цьому notebook не вписане вручну: усі значення, таблиці й підписи походять зі збереженого запуску (`outputs/<mode>/`), позначеного run ID. Стан кожної специфікації («реалізовано», «запущено», «завершено», «пройшла діагностику») визначається за сховищем checkpoint-ів, а не за конфігурацією. Режими `SMOKE`/`PILOT` **не** дають результатів для статті.
""")

md(r"""## 0. Аудит: рецензія ↔ стаття ↔ код (стисло; повна версія — `docs/AUDIT_uk.md`, `docs/REVIEWER_RESPONSE_uk.md`)

Стан на момент редакції v1.1 (актуальний стан кожного експерименту показує таблиця реєстру нижче). «Оцінено» означає: оцінювання завершено в запуску `FULL-2483ee9f` і його результати є на диску; «реалізовано» — код і тести є, оцінювання на повних даних **не** проводилося.

| Код | Зауваження | Що зроблено | Де в коді | Стан |
|---|---|---|---|---|
| A1, A13 | збіжність MCMC, параметри запуску | повна таблиця R̂/ESS/MCSE для гіперпараметрів, пороги, E-BFMI за ланцюгами; «повну збіжність» прибрано; центрована параметризація латентного стану — необов’язковий діагностичний експеримент | `diagnostics.py`, `model.py` (`latent_param`) | частково: E-BFMI < 0,3 в одному ланцюзі основної моделі — подано як застереження (модель із похибками Стьюдента виконує всі критерії й дає близькі оцінки) |
| A2 | додатність ν — наслідок log-зв’язку | у статті це припущення, а не висновок; специфікація з тотожною функцією зв’язку, таблиця знака швидкості, PSIS-LOO | `model.py` (`rate_link="identity"`), `results.sign_table`, `loo.py` | частково: `signed_full` реалізовано, не оцінено |
| A3 | базові моделі та перевірка | МНК з паливною поправкою; без взаємодії; нормальні похибки; фіксоване γ; без ієрархії (прогноз); фіксовані масштаби групових ефектів (v1.2); прогноз сезону 2024; відновлення параметрів | `baseline.py`, `forecast.py`, `recovery.py`, `model.py` (`group_scale="fixed"`), `scripts/run_recovery_study.py` | частково: адаптацію моделі [4] (`ch_adapted_train`) реалізовано, не оцінено; модель із тими самими груповими ефектами без навчених масштабів (`unpooled`, `unpooled_train`, v1.2) реалізовано, не оцінено |
| A4 | апріорні розподіли, P(Soft > Medium > Hard) | таблиця апріорних розподілів формується з коду; усі оцінки параметрів і ймовірність порядку з похибкою Монте-Карло — у таблицях | `model.prior_table`, `results.py`, `paper.py` | усунено |
| A5 | рисунок взаємодії, σ_C за трьома рівнями | n у комірці, ширина інтервалу, шкала без насичення; ефекти компаундів — фіксована шкала (ієрархічний варіант `comp_hier` — чутливість) | `export.fig_interaction`, `model.py` | усунено (рисунок із уніфікованими підписами — після `REPORT_ONLY`) |
| A6 | конструктори проти пілотів | різниця σ_G − σ_D з інтервалом; висновок послаблено | `results.py` | усунено |
| A7 | PPC, чутливість до 1,30 і 5 кіл | усічені репліки, квантилі, PIT; чутливість на рівні даних (точна); повторне оцінювання за інших правил відбору | `ppc.py`, `pipeline.stage_data_sensitivity` | частково: `clean_110`, `minlaps_8` реалізовано, не оцінено (пороги 1,20 і 1,40 змінюють лише 36 і 19 кіл, тому замінені на 1,10 і 1,07) |
| A8, A16, B4, B5, E1–E3 | література, оформлення | список за IEEE, 38 джерел (21 — за 2022–2026 рр., 24 — за 2021–2026 рр.), транслітерований список; дані джерел перевірено | стаття, `docs/REFERENCES_CHECK_uk.md` | усунено (остаточна звірка — за авторами) |
| A9 | відтворюваність | README, ліцензія, `requirements.txt`, seed, `CITATION.cff` | корінь проєкту | частково: DOI архіву — дія авторів |
| A10–A12, A14, A15, B1–B3 | побажання до тексту, рисунків, структури | див. `docs/REVIEWER_RESPONSE_uk.md` | стаття, `export.py` | усунено / частково |

**Межі роботи.** Збережено наукові питання рукопису та компоненти моделі (латентний стан із реальними Δa, ієрархія, обмеження нульової суми, розподіл Гансена). Латентний стан, паливна поправка, скидання на піт-стопі, залежні від компаунда швидкості та асиметричні t-похибки вже є в моделі Cappello–Hoegh — вони **не** оголошуються новими. Нелінійний знос і фізичні пояснення залишено обмеженнями й гіпотезами.
""")

code(r'''# === Конфігурація (єдине місце для налаштувань) ==========================================
MODE = "SMOKE"            # SMOKE | PILOT | FULL | REPORT_ONLY (читає outputs режиму `report_only_source` з config.yaml, без sampling)
CONFIG_YAML = "config.yaml"
FORCE_REFIT = False       # True: видалити fits цього режиму й рахувати заново (нові run ID)
RESUME = True             # True: продовжувати з останнього committed checkpoint

import os, sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = Path.cwd() if (Path.cwd() / "src" / "f1deg").exists() else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import yaml
_raw = yaml.safe_load(open(ROOT / CONFIG_YAML)) or {}
if (_raw.get("sampler") or {}).get("chain_method") == "parallel":   # pmap needs host devices BEFORE jax is imported
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={(_raw.get('sampler') or {}).get('chains', 4)}"
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, pandas as pd
from IPython.display import display, Markdown, Image
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
from f1deg.config import load_config
from f1deg.pipeline import Pipeline
cfg = load_config(ROOT / CONFIG_YAML, mode=MODE, force_refit=FORCE_REFIT, resume=RESUME, data_csv=str(ROOT / _raw.get("data_csv", "data/f1_cache_combined_2022_2024.csv")), outputs_root=str(ROOT / _raw.get("outputs_root", "outputs")))

# Черга оцінювань цього запуску. None — стандартна черга: для FULL це `run_fits` із config.yaml, для SMOKE/PILOT — коротка черга режиму.
# Наступні черги — docs/RUN_PLAN_uk.md, наприклад: RUN_FITS = ["clean_110", "minlaps_8", "ch_adapted_train"].
# Реєстр cfg.experiments (29 специфікацій) НЕ фільтрується: звіти показують і те, що не запускалося;
# уже завершені оцінювання не повторюються (їх знаходять за ідентифікатором).
RUN_FITS = None
if RUN_FITS is not None:
    cfg.fits_override = list(RUN_FITS)

pl = Pipeline(cfg)
print(f"run ID: {pl.run_id} | mode {cfg.mode}{' (REPORT_ONLY: без sampling)' if cfg.report_only else ''} | outputs: {cfg.out_dir}")
print("versions:", json.dumps(cfg.versions()))
print("devices:", jax.devices(), "| x64:", jax.config.jax_enable_x64, "| chain_method:", cfg.sampler.chain_method)
print(f"sampler: {cfg.sampler.chains} chains x ({cfg.sampler.warmup}+{cfg.sampler.draws}), checkpoint every {cfg.sampler.checkpoint_every}, target_accept {cfg.sampler.target_accept}")
print("experiments in queue:", [e.name for e in pl.experiments()])
''')

md(r"""### Реєстр експериментів і їхній фактичний стан

Таблиця нижче будується за сховищем checkpoint-ів (`outputs/<mode>/fits/<fit_id>/`): «завершено» означає, що на диску є всі `warmup + draws` ітерацій кожного ланцюга; «придатний» — що пройдено критерії діагностики або не виконано лише E-BFMI (тоді це зазначено). Специфікація, яку не запускали, має стан «реалізовано, не запущено» незалежно від режиму й черги.
""")

code(r'''# [v1.1] MCMC не запускає
from f1deg import report as RP
_states = pl.states()
_hours = {n: (s["sampling_wall_seconds"] / 3600.0 if s.get("sampling_wall_seconds") else None) for n, s in _states.items()}
display(RP.experiments_status(cfg.experiments, _states, _hours)[["experiment", "пріоритет", "reviewer_points", "in_queue", "статус", "worst_rhat", "min_ess", "ebfmi_min", "sampling_hours", "fit_id"]])
''')

md(r"""## 1. Дані: аудит, очищення, індексація

Правила очищення (у фіксованому порядку, кожен крок логуються): валідний час кола → не `Deleted` → сухі компаунди → без in-/out-laps → без 1-го кола → лише `TrackStatus == 1` → валідні `TyreLife`/`Stint` → ≤ 1.30 × найшвидше чисте коло гонки → стінти з ≥ 5 **збережених** (не обов'язково послідовних) кіл. Визначення віку: `TyreLife` (абсолютний), `set_start_age` (вживані комплекти), `first_retained_age`, `rel_age`, `dage` (реальні Δa). Паливо — детермінований проксі (110 кг, лінійно до `L_r` = макс. номер кола гонки; центрування −55 кг). Команди-перейменування зведено явним `team_map`; пілоти **не** вкладені у конструкторів (crossed-дизайн).
""")

code(r'''data_info = pl.stage_data()
ds = pl.dataset()
display(ds.summary_counts())
display(pd.read_csv(cfg.out_dir / "tables" / "data_cleaning_steps.csv"))
display(pd.read_csv(cfg.out_dir / "tables" / "data_exclusions.csv"))
display(pd.read_csv(cfg.out_dir / "tables" / "data_definitions.csv"))
if cfg.mode != "SMOKE":
    for name in ("stint_integrity", "suspected_shortened", "set_start_age"):
        p = cfg.out_dir / "tables" / f"audit_{name}.csv"
        if p.exists():
            print(f"--- audit_{name}"); display(pd.read_csv(p))
    display(pd.read_csv(cfg.out_dir / "tables" / "data_team_map.csv"))
print("dataset fingerprint:", ds.fingerprint[:16], "| source:", ds.source)
''')

md(r"""## 2. Специфікація моделі (узгоджено зі статтею; повний перелік специфікацій — `docs/MODEL_REGISTRY_uk.md`)

Для кола $n$ відрізка $s$ (траса $q$, компаунд $c$, пілот $d$, конструктор $g$, гонка $r$): відносний вік шин $a_n$ [кіл] від першого збереженого кола відрізка, центрована маса пального $f_n$ [кг], центрований час кола $y_n$ [с].

**Спостереження і латентний стан** (інновації лише для переходів із $\Delta a_n>0$):
$$y_n=\alpha_n+\gamma_{q}f_n+\varepsilon_n,\qquad \alpha_n=\alpha_{n-1}+\nu_s\,\Delta a_n+\sigma_\eta\sqrt{\Delta a_n}\,z_n,\quad z_n\sim N(0,1),\qquad \alpha_{\text{перше коло}}=\beta_s .$$
У нецентрованій параметризації (основна) параметрами є $z_n$: $\alpha_n=\beta_s+\nu_s a_n+\sigma_\eta\sum_{j\le n}\sqrt{\Delta a_j}z_j$. У центрованій (`latent_param="centred"`, необов'язкова діагностична специфікація `primary_centred`) параметрами є самі $\alpha_n$; апостеріорний розподіл той самий, змінюється лише геометрія для sampler.

**Базовий рівень:** $\beta_s=\beta_0+b_{c}+\sigma_\beta z^\beta_s$, $b\sim\mathrm{ZSN}(0{,}7)$ (нормальний розподіл з обмеженням нульової суми, фіксована шкала).

**Швидкість деградації, логарифмічна функція зв'язку (основна модель):**
$$\ln\nu_s=\mu_\nu+C_{c}+T_{q}+D_{d}+G_{g}+I_{qc}+\sigma_U u_s,\qquad u_s\sim N(0,1).$$
$C\sim\mathrm{ZSN}(0{,}5)$ (варіант `comp_hier`: $C=\sigma_C\tilde C$, $\sigma_C\sim HN(0{,}75)$); $T,D,G=\sigma_X\cdot\mathrm{ZSN}(1)$; $I=\sigma_I\cdot\mathrm{ZSN}(1)$ з нульовими сумами за обома індексами. Додатність $\nu_s$ тут — **припущення** моделі, а не висновок із даних.

**Альтернативи для швидкості.** Нове у v1.1 — тотожна функція зв'язку `rate_link="identity"` (специфікація `signed_full`): $\nu_s=\mu^{add}_\nu+C_c+T_q+D_d+G_g+I_{qc}+\sigma^{add}_U u_s$ у с/коло — швидкість може бути від'ємною на кожному рівні — і ефект гонки `race_effect=True`: до лінійного предиктора додається $R_r\sim N(0,\sigma_R^2)$. Варіант `signed_stint` ($\nu_s=\exp(\cdot)+\sigma^{add}_U u_s$: знак вільний лише на рівні відрізка) і центрована параметризація латентного стану були вже у v1.0.

**Фіксовані масштаби групових ефектів** (v1.2, `group_scale="fixed"`, специфікації `unpooled` і `unpooled_train`): той самий лінійний предиктор і ті самі обмеження нульової суми, але $T\sim\mathrm{ZSN}(0{,}5)$, $D,G\sim\mathrm{ZSN}(0{,}3)$, $I\sim\mathrm{ZSN}(0{,}3)$ (за обома індексами) задано безпосередньо, а $\sigma_T,\sigma_D,\sigma_G,\sigma_I$ не оцінюються. Фіксований масштаб дорівнює параметру напівнормального апріорного розподілу відповідного $\sigma$ в основній моделі ($E[\sigma^2]=a^2$), тож апріорна коваріація ефектів в обох моделях однакова; відмінність лише в тому, що основна модель навчає масштаб, тобто силу об'єднання. Порівняння `primary_train` і `unpooled_train` за прогнозом 2024 року відокремлює внесок саме адаптивного об'єднання (таблиця `forecast/forecast_hierarchy_ablation.csv`).

**Паливно-трасовий коефіцієнт:** $\gamma_q=\exp(\mu_\gamma+\sigma_\gamma z^\gamma_q)$ [с/кг] або фіксоване зовнішнє значення 0,03 с/кг (`gamma_fixed`).

**Похибки:** $\varepsilon_n\sim\mathrm{HansenSkewT}(0,\ \sigma_{core},\ \lambda,\ \eta)$, $\eta=2+\tilde\eta$, $\tilde\eta\sim\mathrm{Gamma}(2;\,0{,}5)$; $\sigma_{core}=\sigma_\varepsilon\sqrt{(\eta-2)/\eta}$ залишається скінченним при $\eta\to2$ (за $\lambda=0$ це масштаб розподілу Стьюдента). Варіанти: Стьюдента (`studentt`), нормальні (`gauss`), $\eta\equiv5$ (`skewt_df5`).

**Ідентифіковність ν–γ (твердження 1 і 2 статті).** У межах відрізка $f_n=f_{s,0}-b_r a_n$, де $b_r$ — витрата пального за коло. Тоді $E[y_n]=\theta_s+\kappa_s a_n$ з $\theta_s=\beta_s+\gamma_q f_{s,0}$, $\kappa_s=\nu_s-b_r\gamma_q$: правдоподібність одного відрізка залежить від $(\beta_s,\nu_s,\gamma_q)$ лише через $(\theta_s,\kappa_s)$ і не змінюється за перетворення $\gamma_q\to\gamma_q+\delta$, $\nu_s\to\nu_s+b_r\delta$, $\beta_s\to\beta_s-\delta f_{s,0}$. Отже, дані всередині відрізків **не** розділяють деградацію і паливний тренд. Розділення дає порівняння рівнів відрізків, що починаються з різною масою пального: якщо $E[\beta_s\mid f_{s,0},c]=\beta_0+b_c$ (базовий рівень у середньому не залежить від маси пального на початку відрізка — це реалізує обмінний апріорний розподіл $\beta_s$), то $\gamma_q$ — коефіцієнт регресії $\theta_s$ на $f_{s,0}$ за фіксованого компаунда. Умова змістовна: якщо компаунд пов'язаний із фазою гонки, зсув оцінки $\gamma_q$ переходить у $\nu_s$. Перевірки: синтетичні сценарії `known_gamma`, `phase_confounded`; чутливість `gamma_fixed`, `fuel_prior_wide`.
""")

code(r'''from f1deg.model import prior_table, hyper_sites
display(Markdown("**Таблиця прайорів primary-моделі (згенеровано з коду; це ж джерело використовує модель):**"))
display(prior_table(cfg.primary))
print("dense mass block (гіперпараметри):", hyper_sites(cfg.primary))
display(Markdown("**Матриця експериментів (зафіксована до перегляду результатів):**"))
display(pd.DataFrame([{"name": e.name, "tier": e.tier, "differs": e.differs_from_primary, "data_variant": e.data_variant, "question": e.question} for e in cfg.experiments]))
''')

md(r"""## 3. Перевірка реалізації (likelihood, індексація, resume)

Тести запускаються командою `python -m pytest tests -q` (логи у `tests/logs/`); тести редакції v1.1 лежать у `tests/rev2/` (`python -m pytest tests/rev2 -q`: сумісність ідентифікаторів, нові специфікації, PSIS-LOO, оцінювальні правила, реєстр і звіти). Нижче — короткі inline-перевірки того самого коду: NumPy-reference vs JAX log-density Гансена (різна алгебра), Student-t при λ=0, узгодженість градієнта, PIT вибірки генератора; індексація латентного стану.
""")

code(r'''import jax.numpy as jnp
from scipy import stats as st
from f1deg import hansen as H
y = np.linspace(-8, 8, 2001)
for lam, eta in [(0.5, 5.0), (0.7, 2.05), (-0.4, 3.0)]:
    ref = H.np_logpdf(y, 0.2, 0.3, lam, eta); got = np.asarray(H.jax_logpdf(jnp.asarray(y), 0.2, 0.3, lam, eta))
    print(f"lambda={lam:+.1f} eta={eta}: max|NumPy-ref − JAX| = {np.max(np.abs(ref-got)):.2e}; ∫pdf = {np.trapezoid(np.exp(got), y):.4f}")
print("Student-t limit (lambda=0): max diff =", np.max(np.abs(np.asarray(H.jax_logpdf(jnp.asarray(y), 0.0, 0.3, 0.0, 4.0)) - st.t.logpdf(y, 4.0, 0, 0.3))))
g = jax.grad(lambda p: jnp.sum(H.jax_logpdf(jnp.asarray(y[::40]), p[0], jnp.exp(p[1]), jnp.tanh(p[2]), 2 + jnp.exp(p[3]))))
p0 = jnp.array([0.1, np.log(0.3), np.arctanh(0.6), np.log(2.0)]); ga = np.asarray(g(p0)); h = 1e-6
gn = np.array([(float(jnp.sum(H.jax_logpdf(jnp.asarray(y[::40]), *(lambda q: (q[0], jnp.exp(q[1]), jnp.tanh(q[2]), 2 + jnp.exp(q[3])))(p0 + h*jnp.eye(4)[k])))) - float(jnp.sum(H.jax_logpdf(jnp.asarray(y[::40]), *(lambda q: (q[0], jnp.exp(q[1]), jnp.tanh(q[2]), 2 + jnp.exp(q[3])))(p0 - h*jnp.eye(4)[k]))))) / (2*h) for k in range(4)])
print("gradient: autodiff", ga.round(5), "| finite diff", gn.round(5))
x = np.asarray(H.jax_sample(jax.random.key(1), 0.2, 0.3, 0.5, 4.0, (100000,))); print("sampler PIT KS p-value:", round(st.kstest(H.np_cdf(x, 0.2, 0.3, 0.5, 4.0), "uniform").pvalue, 3))
# індексація латентного стану: інновації тільки при Δa>0, перші кола без інновацій, кумулятивна сума в межах стінта
assert len(ds.inno_idx) == ds.n_laps - ds.n_stints and np.all(ds.dage[ds.first_pos] == 0) and np.all(ds.dage[ds.inno_idx] > 0)
print(f"індексація: {ds.n_laps} кіл, {ds.n_stints} стінтів, {len(ds.inno_idx)} інновацій; Δa>1 у {np.mean(ds.dage[ds.inno_idx] > 1):.1%} переходів; TyreLife/LapNumber mismatch = {ds.laps.attrs.get('tyrelife_lap_mismatch', 0)}")
''')

md(r"""### 3.1 Відновлення після вимкнення електрики — як це працює і як перевірено

* **Backend:** NumPyro low-level `hmc()` (`init_kernel`/`sample_kernel`): кожен перехід — чиста функція `HMCState` (позиція в unconstrained space, градієнт, потенціал, енергія, лічильник `i`, **повний adaptation state**: step size, dual averaging, Welford mass matrix, індекс вікна адаптації, RNG ключі адаптації та ланцюга). `k` переходів одним `lax.scan` + `m` переходів другим = `k+m` одним scan **bitwise** (тест 08).
* **Checkpoint кожні `checkpoint_every` переходів** (warm-up і sampling): chunk draws + повний стан кожного chain → tmp-файл → fsync → rename → commit-record із sha256 усіх файлів → fsync каталогу. Нестійкий хвіст (без commit) відкидається; пошкоджений файл не проходить checksum і run відкочується на попередню generation (зберігаються 2).
* **Lock** із pid/host/heartbeat (stale-lock recovery), **fingerprint** (дані в порядку спостережень, модель+прайори, preprocessing, mapping, конфігурація kernel/adaptation, версії, precision) — несумісний checkpoint не підхоплюється (новий run ID). Збільшення лише `draws` після warm-up продовжує той самий chain (записується в `extensions.json`).
* **Тести у нових процесах** (`tests/test_resume.py`): SIGKILL під час warm-up, SIGKILL під час sampling, crash між записом chunk і commit, пошкоджений останній checkpoint, зміна fingerprint, повторний `Run All`, lock — усі порівнюються з безперервним run **bitwise** (draws, sample_stats, латентні інновації, фінальні стани). Один довгий перехід не можна безпечно перервати — втрачається щонайбільше один chunk.
""")

code(r'''pf = pl.stage_preflight(pl.primary_experiment())
display(pd.DataFrame([{k: v for k, v in pf.items() if k not in ("devices", "restore_mismatch")}]).T.rename(columns={0: "preflight"}))
logs = sorted((ROOT / "tests" / "logs").glob("*.log"))
summ = ROOT / "tests" / "logs" / "resume_tests_summary.json"
if summ.exists():
    display(Markdown("**Підсумок integration-тестів resume (останній запуск pytest):**")); display(pd.DataFrame(json.load(open(summ))))
elif logs:
    print("test logs:", [l.name for l in logs])
for name in ("02b_resume_after_warmup_kill", "05b_resume_after_corruption"):
    p = ROOT / "tests" / "logs" / f"{name}.log"
    if p.exists():
        print(f"--- {p.name} (витяг)"); print("\n".join(l for l in p.read_text().splitlines() if "RESUMING" in l or "completed" in l or "removed" in l)[:1500])
''')

md(r"""## 4. Prior predictive

Спільні наслідки прайорів: розподіл $\nu_s$, накопичене погіршення за 10/20/30 кіл, розкид базових рівнів, паливний тренд за пів гонки, масштаб шуму, квантилі prior-predictive $y$ проти даних.
""")

code(r'''pp = pl.stage_prior_predictive(pl.primary_experiment())
print(json.dumps({k: v for k, v in pp.items() if k not in ("npz",)}, indent=1, ensure_ascii=False))
''')

md(r"""## 5. Fits матриці експериментів (з checkpoint/resume, bounded extension)

Для кожного експерименту: fit (resume з checkpoint, якщо є) → діагностика → gates. Якщо провалені лише ESS/MCSE/R̂-критерії (без дивергенцій), posterior target подвоюється щонайбільше `max_extension_rounds` разів (у FULL/PILOT). Провали залишаються видимими зі статусом `needs_attention`.
""")

code(r'''fit_results = {}
t0 = time.time()
for e in ([] if cfg.report_only else pl.experiments()):
    print(f"\n===== {e.name}: {e.question}")
    fit_results[e.name] = pl.fit_with_bounded_extension(e)
    r = fit_results[e.name]
    if r["diagnostics"]:
        g = r["diagnostics"]["gates"]
        print(f"   status={r['diagnostics']['status']} worst_rhat={g['worst_rhat']:.4f} min_ess={min(g['min_ess_bulk'] or 0, g['min_ess_tail'] or 0):.0f} div={r['diagnostics']['energy']['divergences']} ebfmi_min={r['diagnostics']['energy']['ebfmi_min']:.2f} extension_rounds={r['extension_rounds']}")
    if r["fit"].get("status") == "completed" and e.tier != 3:
        pl.stage_ppc(e)
print(f"\nfits section finished in {(time.time()-t0)/60:.1f} min (this process)")
display(pl.registry.table())
''')

md(r"""## 6. Діагностика MCMC

Операційні критерії (не доказ правильності моделі): rank-normalised split R̂ < 1.01; bulk/tail ESS ≥ 400; MCSE(mean) ≤ 5 % SD; 0 дивергенцій; E-BFMI ≥ 0.3 у кожному chain. Усі значення — неокруглені (`diagnostics/summary_*.csv`), включно з паливними гіперпараметрами, масштабами, взаємодією та латентними блоками (`latent_blocks_*.csv`; thinned ESS занижений).
""")

code(r'''primary = pl.primary_experiment()
dg = pl.registry.result(f"diagnostics_{primary.name}")
if dg:
    print(json.dumps({"gates": dg["gates"], "energy": {k: v for k, v in dg["energy"].items() if k != "tree_depth_hist"}}, indent=1, default=float))
    s = pd.read_csv(dg["summary_csv"])
    cols = [c for c in ("parameter", "mean", "sd", "eti94_lo", "eti94_hi", "rhat", "ess_bulk", "ess_tail", "mcse_rel") if c in s]
    display(s[s["site"].isin(["mu_log_nu", "sigma_track", "sigma_driver", "sigma_team", "sigma_interaction", "sigma_stint", "mu_log_gamma", "sigma_log_gamma", "sigma_eta", "sigma_eps_core", "lambda_skew", "nu_df", "compound_effect"])][cols])
    display(pd.read_csv(dg["latent_csv"]))
''')

md(r"""## 7. Результати, PPC, recovery, прогноз, експорт

Нижче виконуються: OLS baseline, synthetic recovery (незалежний генератор; known vs estimated γ), prefix-only прогноз train→test, а потім експорт усіх рисунків/таблиць/підписів (`figures/`, `tables/`, `figures/captions_uk.json`) і звітів (`reports/`). У `REPORT_ONLY` усе перегенеровується з диска без sampling. Комірка `[v1.1, нове]` після цієї доповнює етапи редакції: PSIS-LOO, чутливість на рівні даних, таблиця знака швидкості, таблиці статті.
""")

code(r'''bl = pl.stage_baseline(); print("OLS baseline:", {k: v for k, v in bl.items() if not k.endswith("csv")})
if not cfg.report_only:
    rec = pl.stage_recovery(); print("recovery:", {k: (v.get("coverage_share_effects") if isinstance(v, dict) else v) for k, v in rec.items()})
    fc = pl.stage_forecast(); print("forecast:", {k: v for k, v in fc.items() if k in ("n_test_stints", "n_target_laps", "train_offset_full_minus_q1_s", "pf_ess_min_quantiles", "note")})
ex = pl.stage_export()
rp = pl.stage_report()
print("figures:", list(ex.get("figures", {}).keys())); print("tables:", len(ex.get("tables", {})))
''')

code(r'''# [v1.1] MCMC не запускає
# Для КОЖНОГО завершеного оцінювання реєстру: діагностика, предиктивні перевірки, PSIS-LOO; далі чутливість
# на рівні даних, прогноз (зокрема логарифмічна предиктивна щільність і схема з коротким початком відрізка),
# рисунки, таблиці статті (outputs/<mode>/paper/) і звіти. Те саме виконує MODE = "REPORT_ONLY".
pl.postprocess()
tdir = cfg.out_dir / "tables"
# forecast_hierarchy_ablation, pooling_effects_*: v1.2, з'являються після оцінювання unpooled_train / unpooled
for name in ("loo_comparison", "data_threshold_sensitivity", "sign_primary", "mcmc_hyper_primary", "forecast_hierarchy_ablation", "pooling_effects_primary_vs_unpooled"):
    p = tdir / f"{name}.csv"
    if p.exists():
        display(Markdown(f"**{name}**")); display(pd.read_csv(p))
if pl.problems:
    display(Markdown("**Етапи, що завершилися помилкою (решту виконано):**")); display(pd.DataFrame(pl.problems, columns=["stage", "error"]))
''')

code(r'''figdir = cfg.out_dir / "figures"; caps = json.load(open(figdir / "captions_uk.json", encoding="utf-8"))
for key in ("data", "ppc", "compound", "tracks", "scales", "interaction", "groups", "examples", "recovery", "sensitivity", "forecast", "forecast_examples", "diagnostics"):
    p = ex.get("figures", {}).get(key)
    if p and Path(p).exists():
        name = Path(p).stem
        display(Markdown(f"**{name}.** {caps.get(name, '')}")); display(Image(filename=p, width=1000))
''')

md(r"""## 8. Таблиці для статті та звіт для редакції
""")

code(r'''tdir = cfg.out_dir / "tables"
for name in ("compound", "compound_contrasts", "order", "scales", "sigma_cmp", "residual", "identification", "gates", "model_comparison", "sensitivity", "forecast", "forecast_pairs", "recovery"):
    p = ex.get("tables", {}).get(name)
    if p and Path(p).exists():
        display(Markdown(f"**{Path(p).stem}**")); display(pd.read_csv(p).head(40))
''')

code(r'''rdir = cfg.out_dir / "reports"
display(Markdown((rdir / "results_summary_uk.md").read_text(encoding="utf-8")) if (rdir / "results_summary_uk.md").exists() else "no summary (primary fit missing)")
if (rdir / "old_conclusions_status.csv").exists():
    display(Markdown("**Статус старих висновків:**")); display(pd.read_csv(rdir / "old_conclusions_status.csv"))
display(Markdown("**Матриця відповіді рецензенту (`reports/reviewer_response_matrix.csv`):**")); display(pd.read_csv(rdir / "reviewer_response_matrix.csv"))
display(Markdown("**Статус експериментів (`reports/experiments_status.csv`):**")); display(pd.read_csv(rdir / "experiments_status.csv")[["experiment", "пріоритет", "in_queue", "статус", "worst_rhat", "min_ess", "ebfmi_min", "sampling_hours", "fit_id"]])
display(Markdown((rdir / "editorial_checklist.md").read_text(encoding="utf-8")))
''')

md(r"""### 8a. Таблиці статті та перелік відсутніх результатів

`outputs/<mode>/paper/` містить таблиці у вигляді, в якому вони стоять у статті (десяткова кома, 94 % інтервали), `paper_numbers.json` з усіма числами тексту та `MISSING.md` — перелік обов'язкових і рекомендованих експериментів, результатів яких ще немає. Таблиця результатів експерименту з'являється лише тоді, коли експеримент завершено.
""")

code(r'''# [v1.1] MCMC не запускає
pdir = cfg.out_dir / "paper"
if (pdir / "MISSING.md").exists():
    display(Markdown((pdir / "MISSING.md").read_text(encoding="utf-8")))
# назви відповідають номерам таблиць статті: 6 — предиктивна перевірка, 7 — порівняння специфікацій, 8 — прогноз, 9 — відновлення параметрів
for name in ("tab2_experiments", "tab3_mcmc", "tab3b_gates", "tab4_compounds", "tab5_scales", "tab6_ppc", "tab7_models", "tab8_forecast", "tab8b_forecast_pairs",
             "tab8d_hierarchy_ablation", "tab9_recovery_study", "tab9_recovery_large", "tabS_sign", "tabS_data_sensitivity"):
    p = pdir / f"{name}.md"
    if p.exists():
        display(Markdown(f"**{name}**\n\n" + p.read_text(encoding="utf-8")))
''')

md(r"""### 8b. Дослідження відновлення параметрів (окремий скрипт, кілька реплік на сценарій)

Запуск `FULL-2483ee9f` містить 24 оцінювання на синтетичних даних (5 сценаріїв × 3 репліки). Ширше дослідження з додатковими сценаріями редакції (`identity_rates`, `identity_zero` — справжні швидкості з тотожною функцією зв'язку, зокрема нульові й від'ємні; `race_effect` — ефект гонки; `phase_confounded` — компаунд, пов'язаний із фазою гонки) запускається окремо, бо кожне оцінювання — самостійний процес із власним checkpoint-ом:

```
python scripts/run_recovery_study.py --scenarios baseline known_gamma identity_rates identity_zero race_effect phase_confounded --start 0 --stop 10 --out outputs_local/recovery_study
python scripts/run_recovery_study.py --merge outputs_local/recovery_study --out outputs_local/recovery_study
```

Перервану команду можна просто повторити. Той самий основний сценарій на утричі більшому синтетичному наборі (перевірка, як зміщення загального рівня залежить від обсягу вибірки):

```
python scripts/run_recovery_study.py --scenarios baseline --start 0 --stop 4 --tracks 10 --seasons 3 --tag _large --out outputs_local/recovery_large
python scripts/run_recovery_study.py --merge outputs_local/recovery_large --out outputs_local/recovery_large --tag _large
```

Результати дослідження, виконаного під час підготовки редакції на скорочених ланцюгах (2 × (300 + 400)), лежать у `results_sandbox/recovery_study/tables/` і `results_sandbox/recovery_study_large/tables/` — це ті числа, що наведені в статті (табл. 9, рис. 8); їх позначено як отримані на скорочених ланцюгах. Поки власного дослідження в `outputs_local/` немає, `REPORT_ONLY` бере для таблиці 9 і рисунка 8 саме ці таблиці й зазначає джерело.
""")

code(r'''# [v1.1]
_out = ROOT / _raw.get("outputs_root", "outputs")
for tag, roots in (("_study", (_out / "recovery_study" / "tables", ROOT / "results_sandbox" / "recovery_study" / "tables")),
                   ("_large", (_out / "recovery_large" / "tables", ROOT / "results_sandbox" / "recovery_study_large" / "tables"))):
    for root in roots:
        p = root / f"synthetic_recovery_summary{tag}.csv"
        if p.exists():
            s = pd.read_csv(p)
            display(Markdown(f"**Відновлення параметрів: `{p.relative_to(ROOT)}`** (coverage — частка справжніх значень у 94 % інтервалі; mean_z ≈ 0 і rms_z ≈ 1 за коректної специфікації; mean_bias — середнє зміщення)"))
            display(s[["scenario", "variant", "well_specified", "parameter", "n", "n_replicates", "coverage", "coverage_se", "mean_z", "rms_z", "mean_bias", "rmse"]])
            break
    else:
        print(f"таблиць дослідження відновлення ({tag.strip('_')}) не знайдено: див. команди вище")
''')

md(r"""## 9. Підсумок запуску

* Стан усіх етапів — у таблиці нижче і в `outputs/<mode>/stages.json`; fits — у `outputs/<mode>/fits/<fit_id>/` (manifest, commit-записи, chunks, стани).
* **FULL:** виставте `MODE = "FULL"` (або `python scripts/run_pipeline.py --mode FULL`), покладіть CSV у `data/`. Орієнтовний час на fit — за PILOT (див. `diagnostics/preflight_*.json` і `logs/pipeline.log`: секунд на перехід після адаптації × (warmup+draws) × chains / паралелізм).
* **Після вимкнення світла:** та сама команда / `Run All`. Нічого вручну відновлювати не потрібно.
* **Лише похідні етапи, рисунки й таблиці (без MCMC):** `MODE = "REPORT_ONLY"` або `python scripts/run_pipeline.py --mode REPORT_ONLY`.
* **Наступні оцінювання** (див. `docs/RUN_PLAN_uk.md`): `python scripts/run_pipeline.py --mode FULL --fits ch_adapted_train clean_110 minlaps_8`, далі `--mode REPORT_ONLY`; стан кожної специфікації — `python scripts/run_pipeline.py --mode FULL --list`.
* **Оновити рисунки в статті** після повторного формування звіту: `python scripts/update_docx_figures.py --docx <стаття.docx> --figures outputs_local/full/figures --out <нова.docx> --drop-notes`.
""")

code(r'''display(pl.registry.table())
display(Markdown("**Події цього запуску (completed / resumed / skipped):**")); display(pd.DataFrame(pl.registry.events))
figs = sorted((cfg.out_dir / "figures").glob("*.pdf")); tabs = sorted((cfg.out_dir / "tables").glob("*.csv")); reps = sorted((cfg.out_dir / "reports").glob("*"))
print(f"exported: {len(figs)} figures (PDF+PNG) in {cfg.out_dir/'figures'}, {len(tabs)} tables (CSV+MD) in {cfg.out_dir/'tables'}, {len(reps)} report files in {cfg.out_dir/'reports'}")
print("figures:", [f.name for f in figs]); print("tables:", [t.name for t in tabs])
print("run ID:", pl.run_id, "| mode:", cfg.mode)
if cfg.mode != "FULL":
    print("FULL не виконано в цьому запуску: результати вище — не для статті.")
''')

nb["cells"] = cells
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0]) if args else ROOT / "f1_tyre_degradation_revision.ipynb"
if out.exists() and "--force" not in sys.argv:
    try:
        old = json.load(open(out, encoding="utf-8"))
        has_outputs = any(c.get("outputs") for c in old.get("cells", []))
    except Exception:
        has_outputs = True
    if has_outputs:
        safe = out.with_suffix(".clean.ipynb")
        print(f"{out.name} містить збережені виходи комірок — не замінюю його (для заміни додайте --force); пишу {safe.name}")
        out = safe
nbf.write(nb, out)
print("wrote", out)
