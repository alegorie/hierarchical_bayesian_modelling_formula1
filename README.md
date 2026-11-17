# f1deg — ієрархічна беєсівська модель простору станів для деградації шин «Формули-1»

Код до статті Ю. Маркова та Д. Таврова «Ієрархічна беєсівська модель простору станів для оцінювання
деградації шин болідів „Формули-1“» (версія після першого рецензування, пакет `f1deg` 1.2).

*English summary.* Reproducible code for a hierarchical Bayesian state-space model of Formula 1 tyre
degradation (seasons 2022–2024, FastF1 lap data): NumPyro/JAX NUTS with exact checkpoint/resume,
MCMC diagnostics, prior/posterior predictive checks, PSIS-LOO, parameter recovery on synthetic data,
leave-future-out forecasts and automatic export of every table and figure of the paper.

## Що тут є

| Шлях | Зміст |
|---|---|
| `src/f1deg/` | пакет: дані, модель, sampler із контрольними точками, діагностика, перевірки, прогноз, експорт |
| `f1_tyre_degradation_revision.ipynb` | notebook: той самий конвеєр із поясненнями та збереженими результатами запуску `FULL-2483ee9f` |
| `scripts/run_pipeline.py` | запуск конвеєра з командного рядка (`--mode`, `--fits`, `--list`, `--extend`, `--stage`) |
| `scripts/run_recovery_study.py` | дослідження відновлення параметрів на синтетичних даних (багато реплік, кілька паралельних процесів, `--merge`) |
| `scripts/update_docx_figures.py` | заміна рисунків у DOCX статті на щойно згенеровані |
| `scripts/build_notebook.py` | генератор чистого notebook (без виходів); виконаний notebook не перезаписує без `--force` |
| `scripts/fit_worker.py`, `scripts/fetch_fastf1.py`, `scripts/make_zip.py` | процес одного оцінювання для тестів відновлення; завантаження даних FastF1; пакування архіву |
| `forecast_diagnostics.py`, `forecast_mc_check.py` | додаткові перевірки прогнозу (вироджені ваги фільтра частинок, похибка Монте-Карло) |
| `config.yaml` | конфігурація (seed, sampler, черга оцінювань) |
| `tests/` | тести: розподіл Гансена, модель, побітове відновлення після переривання, експорт; `tests/rev2/` — тести змін після рецензування |
| `docs/` | підсумок редакції (`SUMMARY_uk.md` — почніть із нього), аудит, реєстр моделей із формулами, матриця відповіді рецензентам, план запусків, перелік змін, перевірка списку літератури |
| `results_sandbox/` | таблиці дослідження відновлення параметрів, виконаного на скорочених ланцюгах 2 × (300 + 400): `recovery_study/` (табл. 9 і рис. 8 статті), `recovery_study_large/` (той самий основний сценарій на утричі більшій вибірці) |
| `data/` | сюди покласти `f1_cache_combined_2022_2024.csv` (дані FastF1; у репозиторій не входять) |

## Встановлення

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Результати статті отримано з Python 3.12.16, NumPy 2.5.3, JAX 0.12.0.1, NumPyro 0.22.0, ArviZ 1.3.0,
pandas 3.0.5, SciPy 1.18.1 (ОС Windows, CPU, подвійна точність). Версії `python`, `numpy`, `jax`,
`numpyro` входять у відбиток кожного оцінювання: з іншими версіями код працює, але вже пораховані
ланцюги не підхоплюються, а рахуються заново.

## Дані

Файл `data/f1_cache_combined_2022_2024.csv` — кола гонок 2022–2024 рр., завантажені бібліотекою
[FastF1](https://github.com/theOehrly/Fast-F1) (74 605 рядків, 68 гонок). Потрібні стовпці перелічено
в `src/f1deg/data.py` (`REQUIRED_COLUMNS`). Умови використання даних визначає їхнє джерело
(FastF1 / офіційний хронометраж Formula 1); у репозиторії даних немає.

## Швидкий старт

```bash
python scripts/run_pipeline.py --mode SMOKE          # синтетичні дані, перевірка всього конвеєра (хвилини)
python -m pytest tests/rev2 -q                       # тести редакції (хвилини; один із них потребує файла даних)
python -m pytest tests -q                            # усі тести, зокрема побітове відновлення після переривання (≈ 15 хв)
F1DEG_RUN_SLOW=1 python -m pytest tests/rev2 -q      # ще й тест жорсткого переривання моделі з фіксованими масштабами
                                                     # (Windows cmd: set F1DEG_RUN_SLOW=1; PowerShell: $env:F1DEG_RUN_SLOW=1)
python scripts/run_pipeline.py --mode FULL --list    # стан кожного експерименту на диску; нічого не запускає
```

Режими: `SMOKE` — синтетичні дані; `PILOT` — три траси, короткі ланцюги (оцінка часу);
`FULL` — усі дані, 4 ланцюги × (2000 адаптації + 2000 збережених ітерацій); `REPORT_ONLY` — без MCMC:
діагностика, предиктивні перевірки, PSIS-LOO, прогноз, таблиці, рисунки та звіти з уже збережених ланцюгів.

Результати режимів `SMOKE` і `PILOT` не є результатами статті.

## Відтворення результатів статті

```bash
python scripts/run_pipeline.py --mode FULL           # черга run_fits із config.yaml
python scripts/run_pipeline.py --mode REPORT_ONLY    # таблиці, рисунки, звіти
```

Вихідні файли: `outputs_local/full/` — `fits/` (ланцюги), `diagnostics/`, `ppc/`, `loo/`, `forecast/`,
`tables/`, `figures/` (PDF і PNG, підписи в `captions_uk.json`), `reports/`, `paper/` (таблиці статті,
`paper_numbers.json`, `MISSING.md` — чого ще бракує). Порядок і вартість запусків — у
`docs/RUN_PLAN_uk.md`; кожна модель із формулами та фактичним станом — у `docs/MODEL_REGISTRY_uk.md`.

Окреме оцінювання додається до вже порахованих так:

```bash
python scripts/run_pipeline.py --mode FULL --fits clean_110
```

Якщо завершене оцінювання не виконує порогів R̂ або ESS, його ланцюги можна продовжити без нової
адаптації (кількість збережених ітерацій подвоюється):

```bash
python scripts/run_pipeline.py --mode FULL --fits noint --extend 1
```

### Чи дає користь саме ієрархічне об’єднання

Специфікації `unpooled` і `unpooled_train` мають той самий лінійний предиктор, що й основна модель, і ті самі
обмеження нульової суми, але ефекти траси, пілота, конструктора та взаємодії отримують **фіксовані** апріорні
масштаби (0,5; 0,3; 0,3; 0,3 — параметри напівнормальних розподілів відповідних σ основної моделі), а
σ_T, σ_D, σ_G, σ_I не оцінюються. Отже, моделі різняться лише тим, чи навчається масштаб, тобто сила об’єднання.

```bash
python scripts/run_pipeline.py --mode FULL --fits unpooled_train unpooled
```

Після завершення `forecast/forecast_hierarchy_ablation.csv` (і `paper/tab8d_hierarchy_ablation.md`) містить парні
різниці MAE, CRPS і LPD за гонками між `pooled_train` (без групових ефектів), `unpooled_train` і `primary_train`,
а `tables/pooling_effects_primary_vs_unpooled.csv` — ті самі ефекти за навчених і за фіксованих масштабів.

### Відновлення параметрів на синтетичних даних

```bash
python scripts/run_recovery_study.py --scenarios baseline known_gamma identity_rates identity_zero race_effect phase_confounded --start 0 --stop 10 --out outputs_local/recovery_study
python scripts/run_recovery_study.py --merge outputs_local/recovery_study --out outputs_local/recovery_study
```

Кілька процесів можна запускати паралельно на різних діапазонах реплік (`--start`, `--stop`) з окремими
`--out`, а потім об’єднати їх через `--merge`. Більший синтетичний набір задають `--tracks` і `--seasons`
(окремий `--out`, таблиці з `--tag _large`). Поки власного дослідження в `outputs_local/` немає,
`REPORT_ONLY` бере таблиці з `results_sandbox/` і зазначає це в підписах.

### Після вимкнення електрики

Запустіть ту саму команду ще раз. Кожні `checkpoint_every` = 25 переходів на диск атомарно записується
повний стан sampler (позиція, градієнт, стан адаптації, генератори випадкових чисел) разом із
контрольними сумами; незавершений хвіст відкидається, завершені етапи пропускаються. Продовжений
ланцюг побітово збігається з безперервним (тести `tests/test_resume.py`).

## Відтворюваність

* Єдине початкове число `seed: 20260610`; з нього SHA-256 виводить окремі числа для кожного ланцюга,
  етапу, репліки та відрізка прогнозу (`config.derive_seed`).
* Відбиток оцінювання = SHA-256 від (дані в порядку спостережень і правила очищення; специфікація
  моделі та апріорних розподілів; налаштування ядра й адаптації; версії бібліотек). Зміна будь-чого з
  цього створює нове оцінювання й не зачіпає старе.
* Нові поля специфікації, додані після рецензування, не входять у відбиток, доки мають значення, за
  якого модель збігається з попередньою, тому вісім оцінювань запуску `FULL-2483ee9f` лишаються чинними
  (`tests/rev2/test_rev2_fingerprint_compat.py`).
* Таблиця апріорних розподілів (табл. 1 статті) генерується функцією `model.prior_table`, яку
  використовує й сама модель.
* Стан кожного експерименту («реалізовано», «запущено», «завершено», «пройшов діагностику»)
  визначається лише за файлами на диску (`reports/experiments_status.csv`).

## Як цитувати

Див. `CITATION.cff`. Постійний ідентифікатор (DOI) версії коду, що відповідає статті, слід створити через
Zenodo після публікації релізу й вписати в `CITATION.cff` та в статтю.

## Ліцензія

Код — за ліцензією MIT (`LICENSE`; остаточний вибір ліцензії за авторами). Дані до складу репозиторію
не входять.
