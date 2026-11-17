
# f1deg — ієрархічна беєсівська модель простору станів для деградації шин «Формули-1»

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24141733.svg)](https://doi.org/10.5281/zenodo.24141733)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Код до статті Ю. Маркова та Д. Таврова «Ієрархічна беєсівська модель простору станів для оцінювання деградації
шин болідів „Формули-1“», поданої до журналу «Вісник НТУ «ХПІ». Серія: Динаміка та міцність машин».
Пакет `f1deg` 1.2.0.

**English summary.** Reproducible code for a hierarchical Bayesian state-space model of Formula 1 tyre degradation,
fitted to FastF1 lap data of the 2022–2024 seasons (57,143 laps, 2,997 stints). Each stint has a latent pace that
drifts with tyre age at a stint-specific degradation rate ν_s; log ν_s is decomposed into compound, track, driver,
constructor and track × compound effects; lap times have Hansen skew-t errors and a track-specific fuel effect.
Sampling uses NumPyro/JAX NUTS with exact checkpoint/resume: a run interrupted by a power cut continues
bit-identically. The pipeline also computes MCMC diagnostics, predictive checks, PSIS-LOO/WAIC, parameter recovery
on synthetic data and a leave-future-out forecast (train 2022–2023, test 2024), and exports every table and figure of
the paper. Quick start: `python scripts/run_pipeline.py --mode SMOKE`. Code: MIT; data: terms of the source
(Formula 1 timing via FastF1).

## Модель коротко

Для кола n відрізка s (траса q, компаунд c, пілот d, конструктор g):

* латентний темп: α_n = β_s + ν_s a_n + σ_η · (випадкове блукання за приростами віку шин), де a_n — відносний вік шин;
* спостереження: y_n ~ skew-t Гансена(α_n + γ_q f_n, σ_core, λ, η), де f_n — центрована маса пального,
  γ_q — паливний коефіцієнт траси;
* швидкість деградації: log ν_s = μ_ν + C_c + T_q + D_d + G_g + I_qc + σ_U u_s з обмеженнями нульової суми.

Повна специфікація, апріорні розподіли й варіанти моделі описано на початку `src/f1deg/model.py`.

## Що тут є

| Шлях | Зміст |
|---|---|
| `src/f1deg/` | пакет: дані, модель, sampler із контрольними точками, діагностика, предиктивні перевірки, PSIS-LOO, прогноз, експорт таблиць і рисунків |
| `scripts/run_pipeline.py` | запуск конвеєра з командного рядка (`--mode`, `--fits`, `--list`, `--extend`, `--stage`) |
| `scripts/run_recovery_study.py` | відновлення параметрів на синтетичних даних (багато реплік, паралельні процеси, `--merge`) |
| `scripts/fetch_fastf1.py` | повторне завантаження даних FastF1 в окремий файл-знімок |
| `scripts/update_docx_figures.py` | заміна рисунків у DOCX статті на щойно згенеровані |
| `scripts/build_notebook.py`, `scripts/fit_worker.py`, `scripts/make_zip.py` | генератор чистого notebook; процес одного оцінювання; пакування архіву |
| `data/` | вхідні дані (розділ «Дані») |
| `f1_tyre_degradation_revision.ipynb` | notebook: той самий конвеєр із поясненнями й виходами запуску `FULL-2483ee9f` |
| `f1_tyre_degradation_hierarchical_ssm.ipynb` | notebook першої версії дослідження |
| `forecast_diagnostics.py`, `forecast_mc_check.py` | додаткові перевірки прогнозу (ваги фільтра частинок, похибка Монте-Карло) |
| `config.yaml` | конфігурація: seed, правила очищення, sampler, черга оцінювань |
| `requirements.txt` | версії бібліотек, з якими отримано результати статті |
| `requirements.lock.txt` | повний перелік версій середовища розробки (Linux, Python 3.13.18), у якому виконано тести й прогін `SMOKE` |
| `experiments_status.csv` | стан і діагностика кожної специфікації в запуску `FULL-2483ee9f` |
| `reviewer_response_matrix.csv` | відповідність зауважень рецензентів змінам у коді та статті |
| `CITATION.cff`, `LICENSE` | опис для цитування; ліцензія MIT |

## Встановлення

```bash
conda create -n f1deg python=3.12.16
conda activate f1deg
pip install -r requirements.txt
```

> **Важливо.** Версії Python, NumPy, JAX і NumPyro входять у відбиток кожного оцінювання. Результати статті
> отримано з Python 3.12.16, NumPy 2.5.3, JAX 0.12.0.1, NumPyro 0.22.0, ArviZ 1.3.0, pandas 3.0.5, SciPy 1.18.1
> (Windows 11, CPU, подвійна точність) — саме ці версії зафіксовано в `requirements.txt`. В іншому середовищі,
> навіть з іншою patch-версією Python, готові ланцюги не знайдуться: `FULL` почне рахувати все заново, а
> `REPORT_ONLY` перезапише звіти порожніми. Перед кожним запуском перевірте `python --version` і
> `python scripts/run_pipeline.py --mode FULL --list`.

## Дані

`data/f1_cache_combined_2022_2024.csv` — кола гонок сезонів 2022–2024, завантажені бібліотекою
[FastF1](https://github.com/theOehrly/Fast-F1): 74 605 рядків, 68 гонок. Після очищення (правила — у `config.yaml`,
розділ `data`) лишається 57 143 кола, 2 997 відрізків, 66 гонок, 25 трас, 28 пілотів і 10 конструкторів; відбиток
набору — `18372bd8ab7362e3`. Навчальна частина для прогнозу (2022–2023) — 36 198 кіл і 1 980 відрізків. Потрібні
стовпці перелічено в `src/f1deg/data.py` (`REQUIRED_COLUMNS`).

`scripts/fetch_fastf1.py` завантажує дані заново в окремий файл-знімок і порівнює його з оригіналом, не
перезаписуючи його. Результати статті отримано з оригінального файла.

Ліцензія MIT поширюється на код. Дані хронометражу належать їхньому джерелу (Formula 1, через FastF1), і умови їх
використання визначає воно.

## Швидкий старт

```bash
python scripts/run_pipeline.py --mode SMOKE          # синтетичні дані: перевірка всього конвеєра (≈ 20 хв)
python scripts/run_pipeline.py --mode FULL --list    # стан кожного оцінювання на диску; нічого не запускає
```

Режими: `SMOKE` — синтетичні дані, короткі ланцюги; `PILOT` — три траси, короткі ланцюги (оцінка часу); `FULL` — усі
дані, 4 ланцюги × (2000 ітерацій адаптації + 2000 збережених); `REPORT_ONLY` — без MCMC: діагностика, предиктивні
перевірки, PSIS-LOO, прогноз, таблиці, рисунки й звіти з уже збережених ланцюгів. Результати `SMOKE` і `PILOT` не є
результатами статті.

## Відтворення результатів статті

Результати статті отримано в запуску `FULL-2483ee9f`: 4 ланцюги × (2000 + 2000), `target_accept` 0,95,
`max_treedepth` 12.

```bash
python scripts/run_pipeline.py --mode FULL                       # черга run_fits із config.yaml
python scripts/run_pipeline.py --mode FULL --fits clean_110      # окреме оцінювання до вже порахованих
python scripts/run_pipeline.py --mode REPORT_ONLY                # таблиці, рисунки, звіти без MCMC
```

Після черги похідні етапи виконуються автоматично. Результати лягають в `outputs_local/full/`: `fits/` (ланцюги),
`diagnostics/`, `ppc/`, `loo/`, `forecast/`, `tables/`, `figures/`, `reports/` (зокрема `experiments_status.csv`) і
`paper/` (таблиці статті та `paper_numbers.json` з усіма числами тексту).

`--extend 1` продовжує ланцюги завершеного оцінювання без нової адаптації. Для навчальних оцінювань це змінює
вибірку, з якої береться прогноз, тому в статті наведено результати неподовжених оцінювань.

### Основні специфікації

| Назва | Відмінність від `primary` |
|---|---|
| `primary` | основна модель |
| `noint` | без взаємодії траса × компаунд |
| `gauss`, `studentt` | нормальні похибки; похибки Стьюдента |
| `gamma_fixed` | паливний коефіцієнт зафіксовано (0,03 с/кг) |
| `unpooled` | ті самі T, D, G, I, але з фіксованими апріорними масштабами (σ_T, σ_D, σ_G, σ_I не оцінюються) |
| `pooled` | без T, D, G, I: лише компаунд і відхилення відрізка |
| `ch_adapted` | адаптація моделі Cappello & Hoegh: log ν_s = μ_ν + C_c |
| `signed_full` | тотожний зв'язок: ν_s у с/коло, знак швидкості не обмежено на жодному рівні |
| `clean_110`, `clean_107`, `minlaps_8`, `minlaps_10` | інші правила відбору кіл |
| `race_effect` | додатковий ефект гонки (траса × сезон) |
| `*_train` | та сама модель на 2022–2023 для прогнозу 2024 |

Усі 29 специфікацій визначено у `src/f1deg/config.py` (`default_experiments`); стан і діагностику кожної наведено в
`experiments_status.csv`.

### Відновлення параметрів на синтетичних даних

```bash
python scripts/run_recovery_study.py --scenarios baseline known_gamma identity_rates identity_zero race_effect phase_confounded --start 0 --stop 10 --out outputs_local/recovery_study
python scripts/run_recovery_study.py --merge outputs_local/recovery_study --out outputs_local/recovery_study
```

Кілька процесів можна запускати паралельно на різних діапазонах реплік (`--start`, `--stop`) з окремими `--out`, а
потім об'єднати через `--merge`.

## Після вимкнення електрики

Запустіть ту саму команду ще раз. Кожні 25 переходів (`checkpoint_every`) на диск атомарно записується повний стан
sampler — позиція, градієнт, стан адаптації, генератори випадкових чисел — разом із контрольними сумами.
Незавершений хвіст відкидається, завершені етапи пропускаються, а продовжений ланцюг побітово збігається з
безперервним.

## Відтворюваність

* Єдине початкове число `seed: 20260610`; з нього через SHA-256 виводяться окремі числа для кожного ланцюга, етапу,
  репліки й відрізка прогнозу (`config.derive_seed`).
* Відбиток оцінювання — SHA-256 від даних і правил очищення, специфікації моделі й апріорних розподілів,
  налаштувань ядра й адаптації та версій бібліотек. Зміна будь-чого з цього створює нове оцінювання й не зачіпає
  старого.
* Поля специфікації, додані до пакета пізніше, не входять у відбиток, доки мають значення, за якого модель
  збігається з попередньою; тому раніше пораховані оцінювання лишаються чинними.
* Таблицю апріорних розподілів статті генерує функція `model.prior_table`, яку використовує й сама модель.
* Стан кожного оцінювання визначається за сховищем контрольних точок на диску; реєстр етапів — лише кеш.

## Як цитувати

Ю. Марков, Д. Тавров. *f1deg: hierarchical Bayesian state-space model of Formula 1 tyre degradation*, версія 1.2.0.
Zenodo, 2026. https://doi.org/10.5281/zenodo.24141733

```bibtex
@software{markov_tavrov_f1deg_2026,
  author    = {Markov, Yurii and Tavrov, Danylo},
  title     = {f1deg: hierarchical Bayesian state-space model of Formula 1 tyre degradation},
  version   = {1.2.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24141733},
  url       = {https://github.com/alegorie/hierarchical_bayesian_modelling_formula1}
}
```

Машиночитний опис для цитування — у `CITATION.cff`.

## Ліцензія

Код поширюється за ліцензією MIT (`LICENSE`). Дані — за умовами їхнього джерела (розділ «Дані»).
