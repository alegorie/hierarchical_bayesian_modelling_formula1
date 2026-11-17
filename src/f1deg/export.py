"""Figures (PDF + PNG 300 dpi) and tables (CSV + Markdown) with captions generated from the persisted run.

No number is typed by hand: every caption string is built from the tables it describes and carries the
run identifier.  Colour convention: Hard grey, Medium ochre, Soft red, plus distinct markers so that
nothing relies on colour alone.

v1.1 (review round 1, remarks A5, A6, A12): all labels are Ukrainian, units on the axes are «с/коло»,
no title overlaps a neighbouring panel (long titles are wrapped), every figure has a complete legend,
the interaction heat map shows n and the interval width in every cell on an unsaturated scale, and the
scales figure contains every component.  Figures that need posterior draws degrade gracefully when only
tables are available (the corresponding panel is omitted, never invented).
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

from .data import COMPOUND_COLORS, COMPOUND_MARKERS, COMPOUND_ORDER

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.titlelocation": "left"})

COMPOUND_UK = {"SOFT": "Soft", "MEDIUM": "Medium", "HARD": "Hard"}
BLUE, ORANGE, GREEN, RED, GREY, BROWN = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8c8c8c", "#8c564b"
STEP_UK = {"valid lap time": "чинний час кола", "not Deleted": "коло не анульовано", "no in-/out-laps": "без кіл заїзду/виїзду з піт-лейн", "drop race lap 1": "без першого кола гонки",
           "green-flag only (TrackStatus==1)": "лише зелений прапор", "valid TyreLife & Stint": "чинні вік шини й номер відрізка"}
MODEL_UK = {"ols_fuel_ext": "МНК + паливна поправка", "primary": "основна", "noint": "без взаємодії", "pooled": "без ієрархії", "ch_adapted": "адаптація моделі Cappello–Hoegh",
            "gauss": "нормальні похибки", "studentt": "похибки Стьюдента", "gamma_fixed": "γ = 0,03 с/кг", "clean_110": "поріг 1,10", "clean_107": "поріг 1,07", "clean_120": "поріг 1,20", "clean_140": "поріг 1,40",
            "minlaps_8": "відрізки ≥ 8 кіл", "minlaps_10": "відрізки ≥ 10 кіл", "signed_full": "тотожний зв'язок", "primary_centred": "центрована параметризація",
            "skewt_df5": "η = 5", "signed_stint": "знакозмінне відхилення відрізка", "race_effect": "з ефектом гонки", "comp_hier": "ієрархічний σ_C",
            "fuel_prior_wide": "ширший апріорний μ_γ", "unpooled": "фіксовані масштаби групових ефектів"}


def model_label(name: str, mark_train: bool = True) -> str:
    """Ukrainian display name of an experiment; training fits are marked."""
    base = name[:-6] if name.endswith("_train") else name
    lab = MODEL_UK.get(base, base)
    return lab + (" (навч. сезони)" if (name.endswith("_train") and mark_train) else "")


def uk(x, nd: int = 2) -> str:
    """Number with a decimal comma."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


_NUM = re.compile(r"^[−-]?\d+\.\d+$")


def decimal_comma(fig) -> None:
    """Replace the decimal point by a comma in every numeric tick label (Ukrainian typography)."""
    fig.canvas.draw()
    for ax in fig.axes:
        for which in ("x", "y"):
            axis = ax.xaxis if which == "x" else ax.yaxis
            labels = [t.get_text() for t in axis.get_ticklabels()]
            if not any(_NUM.match(l) for l in labels):
                continue
            lim = ax.get_xlim() if which == "x" else ax.get_ylim()
            ticks = axis.get_ticklocs()
            axis.set_ticks(ticks)
            axis.set_ticklabels([l.replace(".", ",") if _NUM.match(l) else l for l in labels])
            (ax.set_xlim if which == "x" else ax.set_ylim)(lim)


def step_uk(s: str) -> str:
    s = str(s)
    if s in STEP_UK:
        return STEP_UK[s]
    if s.startswith("dry compound"):
        return "лише сухі компаунди"
    if s.startswith("<="):
        return f"≤ {float(s.split()[1]):.2f} × найшвидше коло гонки".replace(".", ",")
    if s.startswith("stint >="):
        return f"відрізки з ≥ {s.split()[2]} кіл"
    if s.startswith("years in"):
        return "сезони " + s[len("years in "):]
    if s.startswith("tracks in"):
        return "вибрані траси"
    return s


def wrap(s: str, width: int = 46) -> str:
    return "\n".join(textwrap.wrap(s, width))


class Exporter:
    def __init__(self, fig_dir: Path, tab_dir: Path, run_id: str, prob: float):
        self.fig_dir, self.tab_dir, self.run_id, self.prob = Path(fig_dir), Path(tab_dir), run_id, prob
        self.fig_dir.mkdir(parents=True, exist_ok=True); self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.captions: Dict[str, str] = {}
        self.table_captions: Dict[str, str] = {}
        self.cap_path = self.fig_dir / "captions_uk.json"
        if self.cap_path.exists():
            try:
                self.captions = json.load(open(self.cap_path, encoding="utf-8"))
            except Exception:
                self.captions = {}

    # ------------------------------------------------------------ helpers
    def save(self, fig, name: str, caption: str):
        try:
            decimal_comma(fig)
        except Exception:  # pragma: no cover - cosmetic only
            pass
        fig.savefig(self.fig_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig_dir / f"{name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        # the provenance suffix keeps the v1.0 form "94% ETI" (tests/test_export_and_stages.py relies on it)
        self.captions[name] = f"{caption} [запуск {self.run_id}; інтервали — {int(round(self.prob*100))}% ETI, якщо не зазначено інше]"
        json.dump(self.captions, open(self.cap_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return str(self.fig_dir / f"{name}.png")

    def table(self, df: pd.DataFrame, name: str, caption: str, float_fmt: str = "%.4f") -> str:
        df.to_csv(self.tab_dir / f"{name}.csv", index=False, float_format=float_fmt)
        try:
            md = df.to_markdown(index=False, floatfmt=".4f")
        except Exception:
            md = df.to_string(index=False)
        with open(self.tab_dir / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(f"**{caption}** (запуск {self.run_id})\n\n{md}\n")
        self.table_captions[name] = caption
        json.dump(self.table_captions, open(self.tab_dir / "table_captions_uk.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return str(self.tab_dir / f"{name}.csv")

    @property
    def interval_label(self) -> str:
        return f"{int(round(self.prob*100))} % ETI"

    # ------------------------------------------------------------ figures
    def fig_data_overview(self, steps: pd.DataFrame, y: np.ndarray, slopes: pd.DataFrame, by_track: pd.DataFrame, gamma_ext: float) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
        ax = axes[0, 0]
        labels = [step_uk(s) for s in steps["step"]]
        ax.barh(labels, steps["rows"], color=BLUE)
        ax.invert_yaxis(); ax.set_xlabel("кіл залишилося після кроку"); ax.set_title("а) Послідовність очищення даних")
        ax.set_xlim(0, float(steps["rows"].max()) * 1.28)
        for i, (r, s) in enumerate(zip(steps["rows"], steps["retained_share"])):
            ax.text(r, i, f" {int(r):,}".replace(",", " ") + f" ({100*s:.1f} %)".replace(".", ","), va="center", fontsize=7)
        ax = axes[0, 1]
        ax.hist(np.clip(y, -6, 12), bins=120, color=BLUE, density=True)
        ax.set_xlabel("y — час кола мінус медіана гонки, с"); ax.set_ylabel("щільність"); ax.set_title("б) Центрований час кола (очищені кола)")
        ax = axes[1, 0]
        comps = [c for c in COMPOUND_ORDER if c in set(slopes["Compound"])]
        data_raw = [slopes.loc[slopes["Compound"] == c, "raw_slope"].dropna().clip(-0.5, 0.5) for c in comps]
        data_adj = [slopes.loc[slopes["Compound"] == c, "fuel_adj_slope"].dropna().clip(-0.5, 0.5) for c in comps]
        pos = np.arange(len(comps))
        b1 = ax.boxplot(data_raw, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
        b2 = ax.boxplot(data_adj, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
        for b, alpha in ((b1, 0.4), (b2, 0.95)):
            for patch, c in zip(b["boxes"], comps):
                patch.set_facecolor(COMPOUND_COLORS[c]); patch.set_alpha(alpha)
            for med in b["medians"]:
                med.set_color("k")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(pos); ax.set_xticklabels([COMPOUND_UK.get(c, c) for c in comps]); ax.set_ylabel("нахил МНК за відрізком, с/коло")
        ax.set_title(wrap(f"в) Нахили за відрізками: без поправки (світлі) та з паливною поправкою {uk(gamma_ext)} с/кг (насичені)", 60))
        ax = axes[1, 1]
        bt = by_track[by_track["kind"] == "raw"].sort_values("median")
        ax.barh(bt["Location"], bt["median"], color="#bdbdbd", label="без поправки")
        bt2 = by_track[by_track["kind"] != "raw"].set_index("Location").loc[bt["Location"]]
        ax.plot(bt2["median"], bt["Location"], "o", color=RED, ms=4, label=f"з паливною поправкою {uk(gamma_ext)} с/кг")
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("медіанний нахил МНК, с/коло"); ax.set_title("г) Медіанні нахили за трасами")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        return self.save(fig, "fig01_data_overview", "Огляд даних: (а) послідовність очищення; (б) розподіл центрованого часу кола; (в) нахили МНК за відрізками без паливної поправки та з нею "
                         "(поправка — зовнішнє припущення, а не оцінка моделі); (г) медіанні нахили за трасами. Від'ємні нахили без поправки відбивають суміш деградації, вигоряння пального й еволюції траси.")

    def fig_prior_and_ppc(self, prior_npz: str, prior_json: Dict[str, Any], ppc_npz: str, ppc_table: pd.DataFrame, rep_npz: Optional[str], rep_table: Optional[pd.DataFrame], tag: str) -> str:
        pr = np.load(prior_npz); pc = np.load(ppc_npz)
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
        ax = axes[0, 0]
        ctr = 0.5 * (pr["edges"][:-1] + pr["edges"][1:])
        for i, h in enumerate(pr["hist_rep"]):
            ax.plot(ctr, h, color="#9ecae1", lw=0.7, label="апріорні репліки" if i == 0 else None)
        ax.plot(ctr, pr["hist_obs"], color="k", lw=1.6, label="спостережено")
        ax.set_xlim(-10, 20); ax.set_xlabel("y, с"); ax.set_ylabel("щільність"); ax.set_title("а) Апріорна предиктивна перевірка"); ax.legend()
        ax = axes[0, 1]
        lc = 0.5 * (pr["lnu_edges"][:-1] + pr["lnu_edges"][1:])
        ax.plot(lc, pr["lnu_hist"], color=BLUE, label="апріорний розподіл")
        ax.axvspan(np.log(0.02), np.log(0.25), color="orange", alpha=0.2, label="0,02–0,25 с/коло")
        ax.set_xlabel("ln ν_s (ν_s у с/коло)"); ax.set_ylabel("щільність"); ax.set_title("б) Апріорний розподіл швидкості деградації відрізка"); ax.legend()
        ax = axes[0, 2]
        ctr2 = 0.5 * (pc["edges"][:-1] + pc["edges"][1:])
        for i, h in enumerate(pc["hist_rep"]):
            ax.plot(ctr2, h, color="#fcae91", lw=0.7, label="апостеріорні репліки" if i == 0 else None)
        ax.plot(ctr2, pc["hist_obs"], color="k", lw=1.6, label="спостережено")
        ax.set_xlabel("y, с"); ax.set_ylabel("щільність"); ax.set_title(wrap("в) Умовна апостеріорна предиктивна перевірка (репліки усічено як дані)", 44)); ax.legend()
        ax = axes[1, 0]
        ax.bar(np.linspace(0.025, 0.975, 20), pc["pit_hist"], width=0.05, color=BLUE, label="спостережено")
        ax.axhline(0.05, color="k", lw=0.8, ls="--", label="рівномірний розподіл"); ax.set_xlabel("PIT"); ax.set_ylabel("частка кіл")
        ax.set_title("г) Гістограма PIT (умовно на латентні стани)"); ax.legend(loc="lower center")
        ax = axes[1, 1]
        t = ppc_table[ppc_table["replicates"] == "truncated_like_data"].set_index("statistic")
        names = {"q05": "q 0,05", "q50": "медіана", "q95": "q 0,95", "q99": "q 0,99", "iqr": "IQR", "mad": "MAD", "p_gt_3": "P(y>3 с)"}
        stats = [s for s in names if s in t.index]
        xs = np.arange(len(stats))
        ax.errorbar(xs, t.loc[stats, "rep_mean"], yerr=[t.loc[stats, "rep_mean"] - t.loc[stats, "rep_q05"], t.loc[stats, "rep_q95"] - t.loc[stats, "rep_mean"]], fmt="s", color=RED, label="репліки (середнє, 90 %)")
        ax.plot(xs, t.loc[stats, "observed"], "k_", ms=14, mew=2, label="спостережено")
        ax.set_xticks(xs); ax.set_xticklabels([names[s] for s in stats], rotation=20); ax.set_ylabel("с (для P — частка)"); ax.set_title("д) Статистики центру та хвостів"); ax.legend(); ax.set_yscale("symlog", linthresh=0.05)
        ax = axes[1, 2]
        if rep_npz is not None and rep_table is not None:
            for ci, c in enumerate(COMPOUND_ORDER):
                sub = rep_table[(rep_table["compound"] == c) & rep_table["statistic"].str.startswith("slope")]
                if not len(sub):
                    continue
                qs = [int(s.split("q")[1]) for s in sub["statistic"]]
                ax.plot(qs, sub["observed"], marker=COMPOUND_MARKERS[c], color=COMPOUND_COLORS[c], lw=0, ms=6, mec="k", mew=0.4, label=f"{COMPOUND_UK[c]}: спостережено")
                ax.fill_between(qs, sub["rep_q05"], sub["rep_q95"], color=COMPOUND_COLORS[c], alpha=0.25)
                ax.plot(qs, sub["rep_mean"], color=COMPOUND_COLORS[c], lw=1, label=f"{COMPOUND_UK[c]}: репліки (90 %)")
            ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("рівень квантиля, %"); ax.set_ylabel("нахил МНК за відрізком, с/коло")
            ax.set_title(wrap("е) Репліки нових відрізків: квантилі нахилів МНК без поправки", 44)); ax.legend(fontsize=6.5, ncol=1)
        else:
            ax.axis("off")
        fig.tight_layout()
        return self.save(fig, f"fig02_prior_ppc_{tag}", f"Апріорна та апостеріорна предиктивні перевірки моделі «{tag}». (а, б) спільні наслідки апріорних розподілів; (в–д) репліки спостережень "
                         "за оцінених латентних траєкторій, усічені тим самим правилом, що й дані; (е) репліки нових відрізків (нові ефекти відрізків і латентні траєкторії з апостеріорних гіперпараметрів) "
                         "проти емпіричних квантилів нахилів МНК. Вибіркову скошеність не використано як критерій: за індексу хвоста, близького до 2, вона не має скінченної дисперсії.")

    def fig_compounds(self, comp: pd.DataFrame, contrasts: pd.DataFrame, order: pd.DataFrame, rate_draws: Optional[Dict[str, np.ndarray]], tag: str) -> str:
        have_draws = bool(rate_draws)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        cc = comp.set_index("compound").loc[[c for c in COMPOUND_ORDER if c in comp["compound"].values]]
        ys = np.arange(len(cc))
        ax = axes[0]
        if have_draws:
            for c in COMPOUND_ORDER:
                if c in rate_draws:
                    ax.hist(rate_draws[c], bins=60, density=True, color=COMPOUND_COLORS[c], alpha=0.6, label=COMPOUND_UK[c], histtype="stepfilled")
            ax.set_ylabel("апостеріорна щільність"); ax.legend(title="компаунд")
        else:
            ax.errorbar(cc["rate_median_s_per_lap"], ys, xerr=[cc["rate_median_s_per_lap"] - cc["rate_lo"], cc["rate_hi"] - cc["rate_median_s_per_lap"]], fmt="none", ecolor="k", capsize=3)
            for yy, (c, row) in zip(ys, cc.iterrows()):
                ax.plot(row["rate_median_s_per_lap"], yy, marker=COMPOUND_MARKERS[c], color=COMPOUND_COLORS[c], ms=9, mec="k", mew=0.5)
            ax.set_yticks(ys); ax.set_yticklabels([COMPOUND_UK[c] for c in cc.index]); ax.set_ylim(-0.6, len(cc) - 0.4)
        ax.set_xlabel("швидкість деградації, с/коло"); ax.set_title("а) Швидкість деградації за компаундами")
        ax = axes[1]
        if np.isfinite(cc["multiplier_median"]).all():
            ax.errorbar(cc["multiplier_median"], ys, xerr=[cc["multiplier_median"] - cc["multiplier_lo"], cc["multiplier_hi"] - cc["multiplier_median"]], fmt="none", ecolor="k", capsize=3)
            for yy, (c, row) in zip(ys, cc.iterrows()):
                ax.plot(row["multiplier_median"], yy, marker=COMPOUND_MARKERS[c], color=COMPOUND_COLORS[c], ms=9, mec="k", mew=0.5)
                ax.text(row["multiplier_hi"], yy + 0.18, f"n = {int(row['n_stints'])}", va="bottom", ha="right", fontsize=7)
            ax.axvline(1, color="k", lw=0.8, ls="--"); ax.set_xlabel("множник exp(C_c), безрозмірний"); ax.set_title(f"б) Множники компаундів (n — кількість відрізків)")
        else:
            ax.errorbar(cc["effect_median"], ys, xerr=[cc["effect_median"] - cc["effect_lo"], cc["effect_hi"] - cc["effect_median"]], fmt="o", color="k", capsize=3)
            ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_xlabel("адитивний ефект C_c, с/коло"); ax.set_title("б) Ефекти компаундів (тотожний зв'язок)")
        ax.set_yticks(ys); ax.set_yticklabels([COMPOUND_UK[c] for c in cc.index]); ax.set_ylim(-0.6, len(cc) - 0.4)
        ax = axes[2]
        ys = np.arange(len(contrasts))
        ax.errorbar(contrasts["rate_diff_median_s_per_lap"], ys, xerr=[contrasts["rate_diff_median_s_per_lap"] - contrasts["rate_diff_lo"], contrasts["rate_diff_hi"] - contrasts["rate_diff_median_s_per_lap"]],
                    fmt="o", color=BLUE, capsize=3)
        lab = [k.replace("SOFT", "Soft").replace("MEDIUM", "Medium").replace("HARD", "Hard") for k in contrasts["contrast"]]
        ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_yticks(ys); ax.set_yticklabels(lab); ax.set_ylim(-0.6, len(contrasts) - 0.4)
        ax.set_xlabel("різниця швидкостей деградації, с/коло"); ax.set_title("в) Попарні контрасти")
        g = order[order["scope"].str.startswith("global")] if len(order) else order
        if len(g):
            r = g.iloc[0]
            ax.text(0.98, 0.04, f"P(Soft > Medium > Hard) = {r['probability']:.4f}".replace(".", ","), transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        fig.tight_layout()
        return self.save(fig, f"fig03_compounds_{tag}", f"Ефекти компаундів моделі «{tag}»: (а) швидкість деградації на середньому рівні інших чинників; (б) множники exp(C_c) відносно загального рівня з кількістю відрізків; "
                         "(в) попарні різниці швидкостей зі спільних апостеріорних реалізацій та ймовірність порядку.")

    def fig_tracks(self, tr: pd.DataFrame, tag: str) -> str:
        tr = tr.sort_values("rate_median_s_per_lap")
        has_gamma = "gamma_median_s_per_kg" in tr
        has_rel_iv = "relative_rate_lo" in tr
        ncol = 3 if has_gamma else 2
        fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol + 0.6, max(4.2, 0.27 * len(tr) + 1.8)), sharey=False)
        ys = np.arange(len(tr))
        ax = axes[0]
        ax.errorbar(tr["rate_median_s_per_lap"], ys, xerr=[tr["rate_median_s_per_lap"] - tr["rate_lo"], tr["rate_hi"] - tr["rate_median_s_per_lap"]], fmt="o", color=BLUE, capsize=2, ms=4)
        ax.set_yticks(ys); ax.set_yticklabels([f"{t} ({n})" for t, n in zip(tr["track"], tr["n_stints"])], fontsize=7)
        ax.set_xlabel("швидкість деградації, с/коло"); ax.set_title("а) Трасові швидкості\n(у дужках — кількість відрізків)")
        ax = axes[1]
        if has_rel_iv:
            ax.errorbar(tr["relative_rate_pct_per_lap"], ys, xerr=[tr["relative_rate_pct_per_lap"] - tr["relative_rate_lo"], tr["relative_rate_hi"] - tr["relative_rate_pct_per_lap"]], fmt="o", color=GREY, capsize=2, ms=4)
        else:
            ax.plot(tr["relative_rate_pct_per_lap"], ys, "o", color=GREY, ms=4)
        ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlabel("швидкість деградації, % часу кола за коло")
        ax.set_title("б) Відносна швидкість\n(нормування на медіанний час кола)")
        if has_gamma:
            ax = axes[2]
            ax.errorbar(tr["gamma_median_s_per_kg"], ys, xerr=[tr["gamma_median_s_per_kg"] - tr["gamma_lo"], tr["gamma_hi"] - tr["gamma_median_s_per_kg"]], fmt="s", color=BROWN, capsize=2, ms=4)
            ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlabel("γ_q, с/кг")
            ax.set_title("в) Паливно-трасовий коефіцієнт γ_q")
        for a in axes:
            a.set_ylim(-0.7, len(tr) - 0.3); a.grid(axis="y", lw=0.3, alpha=0.5)
        fig.tight_layout()
        return self.save(fig, f"fig04_tracks_{tag}", f"Трасові ефекти моделі «{tag}»: (а) швидкість деградації на трасі за середнього рівня інших чинників; (б) та сама швидкість у відсотках медіанного часу кола траси "
                         "(нормування зменшує, але не усуває залежність від довжини кола); (в) паливно-трасовий коефіцієнт γ_q, який відокремлюється від ν лише за припущень підрозд. 5.4.")

    def fig_scales(self, sc: pd.DataFrame, tag: str) -> str:
        comps = [c for c in dict.fromkeys(sc["component"]) if c not in ("разом", "2×коваріації (перехресний дизайн)")]
        hy = sc[sc["kind"].str.startswith("hyper")].set_index("component"); re_ = sc[sc["kind"].str.startswith("realised")].set_index("component")
        vd = sc[sc["kind"].str.startswith("variance")].set_index("component")
        units = str(sc["units"].iloc[0]) if "units" in sc and len(sc) else "лог с/коло"
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        ax = axes[0]
        ys = np.arange(len(comps))[::-1]
        for k, (tab, col, lab, off) in enumerate(((hy, BLUE, "гіперпараметр σ (масштаб сукупності)", 0.16), (re_, ORANGE, "реалізований SD оцінених ефектів", -0.16))):
            first = True
            for y0, c in zip(ys, comps):
                if c in tab.index:
                    r = tab.loc[c]
                    r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                    ax.errorbar(r["median"], y0 + off, xerr=[[r["median"] - r["lo"]], [r["hi"] - r["median"]]], fmt="o" if k == 0 else "s", color=col, capsize=2, ms=5, label=lab if first else None)
                    first = False
        ax.set_yticks(ys); ax.set_yticklabels(comps); ax.set_xlabel(f"стандартне відхилення, {units}"); ax.set_xlim(left=0)
        ax.set_title("а) Масштаби неоднорідності швидкості деградації"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
        ax = axes[1]
        order = [c for c in comps if c in vd.index]
        if "2×коваріації (перехресний дизайн)" in set(sc["component"]):
            cv = sc[sc["component"] == "2×коваріації (перехресний дизайн)"].iloc[0]
        else:
            cv = None
        ys2 = np.arange(len(order) + (cv is not None))[::-1]
        vals = [vd.loc[c] if not isinstance(vd.loc[c], pd.DataFrame) else vd.loc[c].iloc[0] for c in order]
        med = [v["median"] for v in vals] + ([cv["median"]] if cv is not None else [])
        lo = [v["lo"] for v in vals] + ([cv["lo"]] if cv is not None else [])
        hi = [v["hi"] for v in vals] + ([cv["hi"]] if cv is not None else [])
        ax.barh(ys2, med, xerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], color=[GREEN] * len(order) + ([GREY] if cv is not None else []), capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ys2); ax.set_yticklabels(order + (["2 × коваріації"] if cv is not None else [])); ax.set_xlabel("частка дисперсії лінійного предиктора швидкості між відрізками")
        ax.set_title("б) Розклад дисперсії (з вагами фактичного дизайну)")
        fig.tight_layout()
        return self.save(fig, f"fig05_scales_{tag}", f"Джерела неоднорідності швидкості деградації, модель «{tag}»: (а) гіперпараметри σ та реалізовані стандартні відхилення оцінених ефектів для кожного компонента "
                         "(для компаунда гіперпараметра немає — апріорна шкала фіксована); (б) частки дисперсії лінійного предиктора швидкості між відрізками з урахуванням фактичної кількості відрізків у групах; "
                         "окремий стовпчик — сума коваріацій перехресних ефектів.")

    def fig_interaction(self, it: pd.DataFrame, tag: str) -> str:
        tracks = list(dict.fromkeys(it["track"])); comps = [c for c in COMPOUND_ORDER if c in it["compound"].values]
        M = np.full((len(tracks), len(comps)), np.nan); W = M.copy(); N = M.copy(); F = M.copy()
        additive = not np.isfinite(it["multiplier_median"]).any()
        for _, r in it.iterrows():
            i, j = tracks.index(r["track"]), comps.index(r["compound"])
            M[i, j] = r["effect_median"]; W[i, j] = r["interval_width_log"]; N[i, j] = r["n_stints"]; F[i, j] = r["full_rate_median_s_per_lap"]
        order = np.argsort(np.nanmean(F, axis=1))
        tracks = [tracks[i] for i in order]; M, W, N, F = M[order], W[order], N[order], F[order]
        vmax = float(np.nanmax(np.abs(M)))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, max(5, 0.33 * len(tracks) + 1.8)))
        ax = axes[0]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(tracks)):
            for j in range(len(comps)):
                head = f"{M[i,j]:+.3f}" if additive else f"×{np.exp(M[i,j]):.2f}"
                ax.text(j, i, f"{head}  n={int(N[i,j])}\n±{W[i,j]/2:.2f}".replace(".", ","), ha="center", va="center", fontsize=6, color="k" if abs(M[i, j]) < 0.6 * vmax else "w",
                        fontweight="normal" if N[i, j] >= 5 else "light", fontstyle="normal" if N[i, j] >= 5 else "italic")
        ax.set_xticks(range(len(comps))); ax.set_xticklabels([COMPOUND_UK[c] for c in comps]); ax.set_yticks(range(len(tracks))); ax.set_yticklabels(tracks, fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("ефект взаємодії I_qc, " + ("с/коло" if additive else "лог с/коло"))
        ax.set_title(wrap("а) Взаємодія: " + ("ефект" if additive else "множник exp(I_qc)") + ", кількість відрізків n, ± половина ширини інтервалу", 52))
        ax = axes[1]
        im2 = ax.imshow(F, cmap="viridis", aspect="auto", vmin=np.nanmin(F), vmax=np.nanmax(F))
        mid = 0.5 * (np.nanmin(F) + np.nanmax(F))
        for i in range(len(tracks)):
            for j in range(len(comps)):
                ax.text(j, i, f"{F[i,j]:.3f}".replace(".", ","), ha="center", va="center", fontsize=6.5, color="w" if F[i, j] < mid else "k")
        ax.set_xticks(range(len(comps))); ax.set_xticklabels([COMPOUND_UK[c] for c in comps]); ax.set_yticks(range(len(tracks))); ax.set_yticklabels([])
        cb2 = fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.03); cb2.set_label("швидкість деградації, с/коло")
        ax.set_title(wrap("б) Повна швидкість деградації для пари «траса — компаунд»", 52))
        fig.tight_layout()
        return self.save(fig, f"fig06_interaction_{tag}", f"Взаємодія «траса — компаунд», модель «{tag}»: (а) медіана ефекту взаємодії, кількість відрізків у комірці та половина ширини {self.interval_label} "
                         "у логарифмічній шкалі; кольорова шкала охоплює весь діапазон значень; комірки з n < 5 набрано курсивом (оцінка визначається переважно апріорним розподілом); "
                         "(б) повна швидкість деградації з урахуванням загального рівня, ефектів компаунда й траси та взаємодії.")

    def fig_groups(self, team: pd.DataFrame, driver: pd.DataFrame, sig: pd.DataFrame, tag: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.2, 0.26 * max(len(team), len(driver)) + 1.8)))
        for ax, tab, name, key in ((axes[0], team, "а) Конструктори", "team"), (axes[1], driver, "б) Пілоти", "driver")):
            if not len(tab):
                ax.axis("off"); continue
            additive = not np.isfinite(tab["multiplier_median"]).any()
            col = "effect" if additive else "multiplier"
            tab = tab.sort_values(f"{col}_median")
            ys = np.arange(len(tab))
            ax.errorbar(tab[f"{col}_median"], ys, xerr=[tab[f"{col}_median"] - tab[f"{col}_lo"], tab[f"{col}_hi"] - tab[f"{col}_median"]], fmt="o", color=BLUE, capsize=2, ms=4)
            ax.axvline(0 if additive else 1, color="k", ls="--", lw=0.8)
            ax.set_yticks(ys); ax.set_yticklabels([f"{t} ({n})" for t, n in zip(tab[key], tab["n_stints"])], fontsize=7)
            ax.set_xlabel("адитивний ефект, с/коло" if additive else "множник швидкості деградації, безрозмірний")
            ax.set_title(f"{name} (у дужках — кількість відрізків)")
        if len(sig):
            r = sig.iloc[0]
            fig.text(0.5, -0.01, f"σ_G − σ_D = {r['median']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}];  P(σ_G > σ_D) = {r['P(>0)']:.2f}".replace(".", ","), ha="center", fontsize=9)
        fig.tight_layout()
        return self.save(fig, f"fig07_groups_{tag}", f"Ефекти конструкторів і пілотів, модель «{tag}», із {self.interval_label} та кількістю відрізків. Порівняння масштабів σ_G і σ_D подано апостеріорною різницею, "
                         "а не перекриттям окремих інтервалів. Ефекти описують гоночний темп і можуть бути змішані зі стратегією та трафіком.")

    def fig_diagnostics(self, p, sites: List[str], tag: str) -> str:
        from .paper import SYMBOL

        sites = [s for s in sites if s in p.posterior and p.posterior[s].ndim == 2]
        n = len(sites)
        fig, axes = plt.subplots(n, 2, figsize=(11, 1.5 * n + 1), squeeze=False)
        C = p.n_chains
        for i, s in enumerate(sites):
            a = p.posterior[s]
            for c in range(C):
                axes[i, 0].plot(a[c], lw=0.5, alpha=0.8)
            axes[i, 0].set_ylabel(SYMBOL.get(s, "η − 2" if s == "nu_df_m2" else s), fontsize=8)
            ranks = np.argsort(np.argsort(a.reshape(-1))).reshape(a.shape)
            for c in range(C):
                axes[i, 1].hist(ranks[c], bins=20, histtype="step", lw=1)
            axes[i, 1].axhline(a.shape[1] / 20, color="k", lw=0.6, ls="--")
        axes[0, 0].set_title("Траєкторії ланцюгів"); axes[0, 1].set_title("Рангові гістограми за ланцюгами")
        axes[-1, 0].set_xlabel("ітерація після адаптації"); axes[-1, 1].set_xlabel("ранг у об'єднаній вибірці")
        fig.tight_layout()
        p1 = self.save(fig, f"figS1_trace_rank_{tag}", f"Додаток: траєкторії ланцюгів і рангові гістограми гіперпараметрів моделі «{tag}».")
        E = p.sample_stats["energy"]; dE = np.diff(E, axis=1)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        ax = axes[0]
        ax.hist((E - E.mean(axis=1, keepdims=True)).ravel(), bins=60, density=True, alpha=0.5, label="маргінальна енергія (центрована)")
        ax.hist(dE.ravel(), bins=60, density=True, alpha=0.5, label="зміна енергії між ітераціями")
        ebfmi = (dE ** 2).mean(axis=1) / E.var(axis=1, ddof=1)
        ax.set_title("E-BFMI за ланцюгами: " + "; ".join(f"{v:.3f}".replace(".", ",") for v in ebfmi)); ax.legend(fontsize=7); ax.set_xlabel("енергія")
        mu = "mu_nu_add" if "mu_nu_add" in p.posterior else "mu_log_nu"
        pairs = [(mu, "mu_log_gamma"), ("sigma_eta", "sigma_eps_core")]
        for ax, (x, y) in zip(axes[1:], pairs):
            if x in p.posterior and y in p.posterior:
                div = p.sample_stats["diverging"].reshape(-1).astype(bool)
                ax.scatter(p.flat(x), p.flat(y), s=3, alpha=0.3, c=BLUE)
                if div.any():
                    ax.scatter(p.flat(x)[div], p.flat(y)[div], s=8, c="r", label="дивергентні переходи"); ax.legend()
                ax.set_xlabel(SYMBOL.get(x, x)); ax.set_ylabel(SYMBOL.get(y, y)); ax.set_title(f"кореляція {np.corrcoef(p.flat(x), p.flat(y))[0,1]:.2f}".replace(".", ","))
            else:
                ax.axis("off")
        fig.tight_layout()
        self.save(fig, f"figS2_energy_pairs_{tag}", f"Додаток: енергетична діагностика та спільні апостеріорні розподіли пар параметрів, важливих для ідентифіковності (μ_ν і μ_γ; σ_η і σ_core), модель «{tag}».")
        return p1

    def fig_examples(self, panels: List[Dict[str, Any]], tag: str, kind: str = "smoothing") -> str:
        n = len(panels)
        fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(12.5, 3.9 * int(np.ceil(n / 2))), squeeze=False)
        for ax, pn in zip(axes.ravel(), panels):
            x = pn["age"]
            ax.fill_between(x, pn["pred_lo"], pn["pred_hi"], color="orange", alpha=0.2, label=f"{self.interval_label} для спостереження")
            if kind == "smoothing":
                ax.fill_between(x, pn["fit_lo"], pn["fit_hi"], color=BLUE, alpha=0.35, label=f"{self.interval_label} для α_n + γ_q f_n")
            ax.plot(x, pn["fit_med"], color="navy", lw=1.5, label="медіана")
            c = pn["compound"]
            ax.plot(x, pn["y"], marker=COMPOUND_MARKERS.get(c, "o"), color=COMPOUND_COLORS.get(c, "k"), lw=0, ms=5, mec="k", mew=0.4, label="спостережено")
            if "split" in pn:
                ax.axvline(x[pn["split"] - 1] + 0.5, color="k", ls=":", lw=1)
                ax.text(x[pn["split"] - 1] + 0.7, 0.97, "прогноз →", fontsize=7, va="top", transform=ax.get_xaxis_transform())
            ax.set_title(wrap(pn["title"], 78), fontsize=8); ax.set_xlabel("відносний вік шини, кола"); ax.set_ylabel("y, с")
        axes.ravel()[0].legend(fontsize=6.5, loc="best")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        name = f"fig08_examples_{kind}_{tag}"
        cap = ("Приклади відрізків (згладжування: використано всі кола відрізка): дані, інтервал латентної траєкторії з паливним членом і предиктивний інтервал; правило вибору відрізків наведено в заголовках."
               if kind == "smoothing" else "Приклади прогнозу: параметри відрізка оцінено лише за колами ліворуч від пунктиру, праворуч — прогноз із предиктивним інтервалом.")
        return self.save(fig, name, f"{cap} Модель «{tag}».")

    def fig_recovery(self, rec: pd.DataFrame, nu_pairs: Optional[Tuple[np.ndarray, np.ndarray]], tag: str) -> str:
        from .paper import SYMBOL

        rec = rec.copy(); rec["element"] = rec["element"].fillna("").astype(str)
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
        lo_col = [c for c in rec.columns if c.startswith("eti") and c.endswith("_lo")][0]; hi_col = lo_col.replace("_lo", "_hi")
        ax = axes[0, 0]
        sc = rec[(rec["element"] == "") & ~rec["parameter"].str.contains("summary")].reset_index(drop=True)
        ys = np.arange(len(sc))
        z = (sc["post_median"] - sc["truth"]) / sc["post_sd"]
        ax.errorbar(z, ys, xerr=[(sc["post_median"] - sc[lo_col]) / sc["post_sd"], (sc[hi_col] - sc["post_median"]) / sc["post_sd"]], fmt="o", color=BLUE, capsize=2, ms=4)
        ax.axvline(0, color="k", lw=0.8); ax.set_yticks(ys); ax.set_yticklabels([SYMBOL.get(s.split(" ")[0], s) for s in sc["parameter"]], fontsize=7)
        ax.set_xlabel("(апостеріорна медіана − істина) / SD")
        ax.set_title(f"а) Скалярні параметри: істина в інтервалі для {int(sc['covered'].sum())} з {len(sc)}")
        ax = axes[0, 1]
        names = {"track_effect": "траса", "compound_effect": "компаунд", "team_effect": "конструктор", "driver_effect": "пілот", "interaction_effect": "траса×компаунд", "compound_base_effect": "базовий рівень компаунда"}
        vec = rec[(rec["element"] != "") & rec["parameter"].isin(list(names))]
        for name, g in vec.groupby("parameter"):
            ax.errorbar(g["truth"], g["post_median"], yerr=[g["post_median"] - g[lo_col], g[hi_col] - g["post_median"]], fmt="o", ms=3, capsize=1, label=f"{names[name]} ({int(g['covered'].sum())}/{len(g)})", alpha=0.8)
        lim = [vec["truth"].min() - 0.1, vec["truth"].max() + 0.1] if len(vec) else [-1, 1]
        ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("істинний ефект"); ax.set_ylabel("апостеріорна медіана"); ax.legend(fontsize=6.5, title="в інтервалі / усього", title_fontsize=7)
        ax.set_title("б) Групові ефекти")
        ax = axes[1, 0]
        g = rec[rec["parameter"] == "gamma_track"]
        if len(g):
            ax.errorbar(g["truth"], g["post_median"], yerr=[g["post_median"] - g[lo_col], g[hi_col] - g["post_median"]], fmt="s", color=BROWN, capsize=2)
            lim = [min(g["truth"].min(), g[lo_col].min()) * 0.9, max(g["truth"].max(), g[hi_col].max()) * 1.1]
            ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("істинний γ_q, с/кг"); ax.set_ylabel("апостеріорна медіана γ_q, с/кг")
            ax.set_title(f"в) Паливно-трасовий коефіцієнт: істина в інтервалі для {int(g['covered'].sum())} з {len(g)}")
        else:
            ax.text(0.5, 0.5, "γ у цій моделі зафіксовано", ha="center"); ax.set_title("в) Паливно-трасовий коефіцієнт"); ax.axis("off")
        ax = axes[1, 1]
        if nu_pairs is not None:
            ax.scatter(nu_pairs[0], nu_pairs[1], s=8, alpha=0.5, color=GREEN)
            lo_, hi_ = min(nu_pairs[0].min(), nu_pairs[1].min(), 0.0), max(nu_pairs[0].max(), nu_pairs[1].max())
            ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.8); ax.set_xlabel("істинна ν_s, с/коло"); ax.set_ylabel("апостеріорна медіана ν_s, с/коло")
            r = np.corrcoef(nu_pairs[0], nu_pairs[1])[0, 1]
            ax.set_title(f"г) Швидкості деградації відрізків: кореляція {r:.2f}".replace(".", ","))
        else:
            ax.axis("off")
        fig.tight_layout()
        return self.save(fig, f"figS3_recovery_{tag}", f"Додаток: відновлення параметрів на синтетичних даних ({tag}); генератор даних не має спільного коду з моделлю. Одна репліка — приклад; зведення за репліками подано в таблиці відновлення.")

    def fig_recovery_summary(self, summ: pd.DataFrame, label: str = "", note: str = "") -> Optional[str]:
        """Coverage and standardised bias of the link-free quantities over replicates, per scenario / fitted model.

        Filled circles: the fitted model contains the generating process; open squares: it does not (deliberate
        misspecification).  The legend is below the panels so that it never covers the points."""
        keep = ["rate_global (s/lap)", "rate_SOFT (s/lap)", "rate_MEDIUM (s/lap)", "rate_HARD (s/lap)", "rate_SOFT_minus_HARD (s/lap)", "gamma_track", "sigma_eta", "sigma_eps_core",
                "nu_stint (per-stint rates)"]
        names = {"rate_global (s/lap)": "загальний рівень", "rate_SOFT (s/lap)": "швидкість Soft", "rate_MEDIUM (s/lap)": "швидкість Medium", "rate_HARD (s/lap)": "швидкість Hard",
                 "rate_SOFT_minus_HARD (s/lap)": "Soft − Hard", "gamma_track": r"$\gamma_q$ (траси)", "sigma_eta": r"$\sigma_\eta$", "sigma_eps_core": r"$\sigma_{\mathrm{core}}$",
                 "nu_stint (per-stint rates)": r"$\nu_s$ (відрізки)"}
        scen_uk = {"baseline": "основний", "known_gamma": "відомий γ", "weak_signal": "слабкий сигнал", "no_interaction": "без взаємодії", "signed_rates": "знакозмінні відхилення відрізків",
                   "heavy_tails": "важкі хвости", "identity_rates": "тотожний зв'язок, рівень 0,05", "identity_zero": "тотожний зв'язок, рівень 0",
                   "race_effect": "ефект траси змінюється між сезонами", "phase_confounded": "компаунд пов'язаний із порядком відрізків"}
        var_uk = {"est": "основна модель", "ident": "модель із тотожним зв'язком", "fixed": "фіксований γ", "race": "модель з ефектом гонки", "noint": "модель без взаємодії",
                  "signed": "модель зі знакозмінним відхиленням"}
        d = summ[summ["parameter"].isin(keep)].copy()
        if not len(d):
            return None
        d["group"] = [f"{scen_uk.get(sc, sc)}: {var_uk.get(v, v)}" + ("" if w else " (не містить породжувального процесу)") for sc, v, w in zip(d["scenario"], d["variant"], d["well_specified"])]
        groups = list(dict.fromkeys(d["group"]))
        well = {gname: bool(d.loc[d["group"] == gname, "well_specified"].iloc[0]) for gname in groups}
        rows_leg = int(np.ceil(len(groups) / 2))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.4, 0.42 * len(keep) + 1.4) + 0.26 * rows_leg), sharey=True)
        ys = np.arange(len(keep))[::-1]
        cmap = plt.get_cmap("tab10")
        handles = []
        for gi, gname in enumerate(groups):
            g = d[d["group"] == gname].set_index("parameter")
            off = (gi - (len(groups) - 1) / 2) * 0.8 / max(len(groups), 1)
            y = [yy + off for yy, k in zip(ys, keep) if k in g.index]
            kk = [k for k in keep if k in g.index]
            col = cmap(gi % 10)
            style = dict(marker="o", mfc=col) if well[gname] else dict(marker="s", mfc="white")
            axes[0].errorbar(g.loc[kk, "coverage"], y, xerr=1.88 * g.loc[kk, "coverage_se"].fillna(0), ls="none", ms=4.5, capsize=2, color=col, mec=col, elinewidth=0.9, **style)
            h, = axes[1].plot(g.loc[kk, "mean_z"], y, ls="none", ms=4.5, color=col, mec=col, **style)
            handles.append((h, gname))
        for yy in ys[:-1]:
            for ax in axes:
                ax.axhline(yy - 0.5, color="0.85", lw=0.5)
        axes[0].axvline(self.prob, color="k", ls="--", lw=0.8); axes[0].set_xlim(-0.02, 1.04)
        axes[0].set_xlabel(f"частка інтервалів, що містять істинне значення (номінал {self.prob:.2f})".replace(".", ","))
        axes[0].set_title("а) Покриття інтервалів"); axes[0].set_yticks(ys); axes[0].set_yticklabels([names.get(k, k) for k in keep]); axes[0].set_ylim(-0.6, len(keep) - 0.4)
        axes[1].axvline(0, color="k", ls="--", lw=0.8); axes[1].set_xlabel("середнє (істинне значення − апостеріорне середнє) / SD"); axes[1].set_title("б) Стандартизоване відхилення")
        fig.legend([h for h, _ in handles], [t for _, t in handles], loc="lower center", ncol=2, fontsize=7.5, frameon=False, title="сценарій: оцінювана модель", title_fontsize=8)
        fig.tight_layout(rect=(0, 0.035 + 0.033 * rows_leg, 1, 1))
        return self.save(fig, f"figS3b_recovery_summary{label}", "Відновлення параметрів на синтетичних даних за репліками: (а) частка інтервалів, що містять істинне значення (відрізки — наближений "
                         "біноміальний інтервал; елементи однієї репліки корельовані); (б) середнє стандартизоване відхилення. Порожні квадрати — моделі, що навмисно не містять породжувального "
                         "процесу: вони показують наслідки хибної специфікації." + note)

    def fig_sensitivity(self, sens: pd.DataFrame, tag: str = "all") -> str:
        """sens: columns model, quantity, median, lo, hi"""
        qs = list(dict.fromkeys(sens["quantity"]))
        fig, axes = plt.subplots(1, len(qs), figsize=(3.3 * len(qs) + 1.2, max(3.6, 0.3 * sens["model"].nunique() + 1.6)), squeeze=False)
        models = list(dict.fromkeys(sens["model"]))
        for k, (ax, q) in enumerate(zip(axes[0], qs)):
            g = sens[sens["quantity"] == q].set_index("model")
            ys = np.array([len(models) - 1 - models.index(m) for m in g.index])
            ax.errorbar(g["median"], ys, xerr=[g["median"] - g["lo"], g["hi"] - g["median"]], fmt="o", color=BLUE, capsize=2, ms=4)
            ax.set_yticks(range(len(models))); ax.set_yticklabels([model_label(m) for m in models[::-1]] if k == 0 else [], fontsize=7); ax.set_ylim(-0.6, len(models) - 0.4)
            ax.set_title(wrap(q, 26), fontsize=8.5); ax.grid(axis="y", lw=0.3, alpha=0.5)
        fig.tight_layout()
        return self.save(fig, f"figS4_sensitivity_{tag}", "Чутливість основних оцінок до специфікації моделі (кожен рядок — окреме оцінювання; діагностичний стан — у таблиці порівняння моделей).")

    def fig_forecast(self, summary: pd.DataFrame, pairs: pd.DataFrame, tag: str = "", reference: str = "primary_train") -> str:
        s = summary[summary["scheme"] == "fixed_origin"].reset_index(drop=True)
        has_lpd = "LPD_race_mean" in s and np.isfinite(s["LPD_race_mean"]).any()
        cols = [("MAE_race_mean", "MAE_race_se", "MAE, с"), ("CRPS_race_mean", "CRPS_race_se", "CRPS, с")] + ([("LPD_race_mean", "LPD_race_se", "LPD (більше — краще)")] if has_lpd else [])
        fig, axes = plt.subplots(1, len(cols) + 1, figsize=(4.0 * (len(cols) + 1) + 0.5, 3.9))
        ys = np.arange(len(s))[::-1]
        labels = [model_label(m, mark_train=False) for m in s["model"]]
        for k, (ax, (m, se, lab)) in enumerate(zip(axes, cols)):
            tq = st.t.ppf(0.5 + self.prob / 2, np.maximum(s["n_races"] - 1, 1))
            ax.errorbar(s[m], ys, xerr=tq * s[se].fillna(0), fmt="o", color=BLUE, capsize=3)
            ax.set_yticks(ys); ax.set_yticklabels(labels if k == 0 else [], fontsize=7.5); ax.set_ylim(-0.6, len(s) - 0.4)
            ax.set_xlabel(lab); ax.set_title(wrap(f"{'абв'[k]}) {lab.split(',')[0].split(' (')[0]}: середнє за гонками", 34)); ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax = axes[-1]
        ax.barh(ys + 0.2, s["coverage94"], height=0.38, label=f"інтервал {int(round(self.prob*100))} %", color=GREEN); ax.barh(ys - 0.2, s["coverage50"], height=0.38, label="інтервал 50 %", color=ORANGE)
        ax.axvline(self.prob, color=GREEN, ls="--", lw=0.9); ax.axvline(0.5, color=ORANGE, ls="--", lw=0.9)
        ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlim(0, 1); ax.set_xlabel("частка кіл усередині інтервалу"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        ax.set_title(f"{'абвг'[len(cols)]}) Покриття предиктивних інтервалів")
        fig.tight_layout()
        return self.save(fig, f"figS5_forecast{tag}", "Прогнозна перевірка: навчання на попередніх сезонах, прогноз останніх кіл відрізків тестового сезону лише з їхнього початку (фіксований початок прогнозу). "
                         f"Середні за гонками з {self.interval_label.replace('ETI', 'довірчим інтервалом (t-розподіл; одиниця вибірки — гонка)')}; покриття предиктивних інтервалів проти номіналу (пунктир).")
