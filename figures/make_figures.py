"""results/*.json -> figures/*.pdf + *.png and paper/results.tex (all numbers in the paper).

Usage: python figures/make_figures.py [--bench results/rtx3070-laptop.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = 3.4
FULL = 7.0
COLORS = {"numpy": "#7f7f7f", "torch-eager": "#9467bd", "torch-compile": "#8c564b",
          "triton": "#1f77b4", "warp": "#2ca02c", "numba": "#ff7f0e", "cupy-cuda-c": "#d62728"}
ORDER = list(COLORS)
LABEL = {"numpy": "NumPy", "torch-eager": "torch eager", "torch-compile": "torch.compile",
         "triton": "Triton", "warp": "Warp", "numba": "Numba-CUDA", "cupy-cuda-c": "CUDA C (CuPy)"}
CEIL_STYLE = {"color": "0.25", "lw": 1.0, "ls": "--"}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def pick_bench(path: str | None) -> Path:
    if path:
        return Path(path)
    for name in ("rtx3070-laptop.json", "provisional.json", "dev-quick.json"):
        p = ROOT / "results" / name
        if p.exists():
            return p
    raise SystemExit("no benchmark JSON in results/; run bench/run.py first")


def label(run: dict) -> str:
    v = run["variant"]
    return f"{LABEL[run['backend']]} ({v})" if v != "int8" else LABEL[run["backend"]]


def runs_2d_metro(bench: dict) -> list[dict]:
    return [r for r in bench["runs"] if r["dim"] == 2 and r["algorithm"] == "metropolis"]


def best_per_variant(runs: list[dict]) -> dict[tuple[str, str], dict]:
    best: dict[tuple[str, str], dict] = {}
    for r in runs:
        k = (r["backend"], r["variant"])
        if k not in best or r["median"] > best[k]["median"]:
            best[k] = r
    return best


# ------------------------------------------------------------------------------ performance --
def fig_ladder(bench: dict) -> None:
    """Best flips/ns per backend/variant, int8 and multi-spin panels, with both ceilings."""
    best = best_per_variant(runs_2d_metro(bench))
    c = bench["ceiling"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [1, 1]})
    for ax, stream, title in ((axes[0], "int8", "int8 spins, 3 B/update"),
                              (axes[1], "msc", "1-bit multi-spin coding, 0.375 B/update")):
        rows = [r for k, r in best.items() if (r["dtype"] == "int8") == (stream == "int8")]
        rows.sort(key=lambda r: (ORDER.index(r["backend"]), r["variant"]))
        y = np.arange(len(rows))[::-1]
        vals = [r["median"] for r in rows]
        ax.barh(y, vals, color=[COLORS[r["backend"]] for r in rows], height=0.62)
        for yi, r in zip(y, rows, strict=True):
            ax.text(r["median"] * 1.03, yi, f"{r['median']:.1f}  ({r['pct_copy_bw']:.0f}%)",
                    va="center", fontsize=7, color="0.15")
        ax.set_yticks(y, [label(r) for r in rows], fontsize=7.5)
        mem = c["copy_bandwidth_GBs"] / rows[0]["bytes_per_unit"] if rows else 1
        ax.axvline(mem, **CEIL_STYLE)
        ax.text(mem * 0.93, -0.55, f"copy-BW ceiling {mem:.0f}", fontsize=6.5, va="bottom",
                ha="right", color="0.25")
        ax.set_xscale("log")
        ax.set_xlabel("flips / ns (median at best size)")
        ax.set_ylim(-0.8, len(rows) - 0.4)
        ax.set_title(title, fontsize=9)
        lo = min(vals + [mem]) / 3
        ax.set_xlim(lo, mem * 3.5)
    fig.tight_layout()
    save(fig, "ladder")


def fig_scaling(bench: dict) -> None:
    """flips/ns vs lattice size, per backend, with the memory ceilings."""
    runs = runs_2d_metro(bench)
    c = bench["ceiling"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), sharey=True)
    for ax, packed, title in ((axes[0], False, "int8"), (axes[1], True, "multi-spin coding")):
        series = defaultdict(list)
        for r in runs:
            if (r["dtype"] != "int8") == packed:
                series[(r["backend"], r["variant"])].append(r)
        for (be, var), rs in sorted(series.items(), key=lambda kv: ORDER.index(kv[0][0])):
            rs.sort(key=lambda r: r["size"][0])
            ls = ":" if ("rng-per-site" in var or "eager32" in var) else "-"
            ax.plot([r["size"][0] ** 2 for r in rs], [r["median"] for r in rs], ls, marker="o",
                    ms=3, lw=1.3, color=COLORS[be], label=label(rs[0]))
        bpu = 3.0 / 8.0 if packed else 3.0
        ax.axhline(c["copy_bandwidth_GBs"] / bpu, **CEIL_STYLE)
        ax.text(1.2e6, c["copy_bandwidth_GBs"] / bpu * 1.08, "copy-bandwidth ceiling",
                fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("spins $N = L^2$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("flips / ns (median)")
    fig.tight_layout()
    save(fig, "scaling")


def fig_roofline(bench: dict) -> None:
    """Classic roofline: flips/ns vs spin updates per DRAM byte, bandwidth slope + RNG roofs."""
    c = bench["ceiling"]
    best = best_per_variant(runs_2d_metro(bench))
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    x = np.logspace(-1.2, 1.3, 100)
    ax.plot(x, c["copy_bandwidth_GBs"] * x, **CEIL_STYLE)
    ax.plot(x, c["spec_bandwidth_GBs"] * x, color="0.6", lw=0.8, ls="--")
    ax.text(1.3, c["copy_bandwidth_GBs"] * 1.3 * 0.22, f"copy {c['copy_bandwidth_GBs']:.0f} GB/s",
            rotation=33, fontsize=6.5, ha="center")
    ph = c["philox_calls_per_ns"]
    cpw = c.get("msc_philox_calls_per_word_warp_max_at_Tc",
                c["msc_philox_calls_per_word_at_Tc"])
    for cps, name, xr in ((0.25, "Philox roof, 4 sites/call", (0.06, 0.6)),
                          (cpw / 64, "Philox roof, lazy bits (warp)", (0.6, 20))):
        ax.hlines(ph / cps, *xr, color="0.4", lw=0.9, ls=":")
        ax.text(xr[0] * 1.05, ph / cps * 1.1, name, fontsize=6.5, color="0.3")
    for r in best.values():
        ax.plot(1 / r["bytes_per_unit"], r["median"], "o", ms=4.5, color=COLORS[r["backend"]],
                mec="white", mew=0.6, alpha=0.9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[b], label=LABEL[b])
               for b in ORDER if any(k[0] == b for k in best)]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("spin updates per DRAM byte")
    ax.set_ylabel("flips / ns")
    ax.set_ylim(0.05, 4000)
    save(fig, "roofline")


# ------------------------------------------------------------------------------- validation --
def fig_validation(val: dict) -> None:
    from pyising.exact import TC_2D, energy_2d, energy_torus, magnetization_2d

    rows = val["scan2d"]
    L = rows[0]["L"]
    Ts = np.linspace(1.5, 3.5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    ax = axes[0]
    ax.plot(Ts, magnetization_2d(Ts), color="k", lw=1.0, label="Onsager-Yang ($L=\\infty$)")
    style = {("int8", "metropolis"): ("o", "Metropolis int8"),
             ("msc", "metropolis"): ("s", "Metropolis multi-spin"),
             ("int8", "heatbath"): ("^", "heat bath int8")}
    shades = {"int8metropolis": "#d62728", "mscmetropolis": "#1f77b4", "int8heatbath": "#2ca02c"}
    for (var, alg), (mk, lab) in style.items():
        rs = [r for r in rows if r["variant"] == var and r["algorithm"] == alg]
        ax.errorbar([r["T"] for r in rs], [r["abs_m"] for r in rs],
                    yerr=[r["abs_m_err"] for r in rs], fmt=mk, ms=3.2, mfc="none", lw=0.8,
                    color=shades[var + alg], label=f"{lab}, $L={L}$")
    ax.axvline(TC_2D, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("$T$")
    ax.set_ylabel(r"$\langle|m|\rangle$")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    ax.plot(Ts, energy_2d(Ts), color="0.6", lw=1.0, label="Onsager ($L=\\infty$)")
    Tf = np.linspace(1.5, 3.5, 120)
    ax.plot(Tf, [energy_torus(t, L) for t in Tf], color="k", lw=1.0, ls="--",
            label=f"exact {L}$\\times${L} torus")
    for (var, alg), (mk, lab) in style.items():
        rs = [r for r in rows if r["variant"] == var and r["algorithm"] == alg]
        ax.errorbar([r["T"] for r in rs], [r["e"] for r in rs], yerr=[r["e_err"] for r in rs],
                    fmt=mk, ms=3.2, mfc="none", lw=0.8, color=shades[var + alg], label=lab)
    ax.set_xlabel("$T$")
    ax.set_ylabel("$e = E/N$")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ins = ax.inset_axes([0.13, 0.62, 0.36, 0.33])
    for (var, alg), (mk, _) in style.items():
        rs = [r for r in rows if r["variant"] == var and r["algorithm"] == alg]
        ins.errorbar([r["T"] for r in rs], [r["e"] - r["e_exact_L"] for r in rs],
                     yerr=[r["e_err"] for r in rs], fmt=mk, ms=2.2, mfc="none", lw=0.6,
                     color=shades[var + alg])
    ins.axhline(0, color="k", lw=0.6)
    ins.set_ylabel("$e - e_{exact}$", fontsize=6)
    ins.tick_params(labelsize=5.5)
    fig.tight_layout()
    save(fig, "validation")


def fig_binder(val: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    greys = plt.cm.viridis
    for ax, key in zip(axes, ("binder2d", "binder3d"), strict=True):
        b = val[key]
        Ts = np.array(b["T"])
        for i, L in enumerate(b["L"]):
            col = greys(i / max(1, len(b["L"]) - 1) * 0.85)
            ax.errorbar(Ts, b["U"][str(L)], yerr=b["U_err"][str(L)], fmt="o-", ms=2.8, lw=1.0,
                        color=col, label=f"$L={L}$")
        ax.axvline(b["Tc_reference"], color="k", lw=0.8, ls="--")
        ref = "Onsager" if b["dim"] == 2 else "Ferrenberg et al."
        ax.text(b["Tc_reference"], ax.get_ylim()[0], f" $T_c$ ({ref})", fontsize=6.5,
                va="bottom")
        ax.set_xlabel("$T$")
        ax.set_ylabel("$U_4$")
        ax.set_title(f"{b['dim']}D, Wolff; last crossing $T={b['Tc_estimate']:.4f}$",
                     fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "binder")


# ------------------------------------------------------------------------------ results.tex --
def fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def write_results_tex(bench: dict, val: dict | None, bench_path: Path) -> None:
    runs = runs_2d_metro(bench)
    best = best_per_variant(runs)
    top = max(best.values(), key=lambda r: r["median"])
    top_py = max((r for r in best.values() if r["backend"] != "cupy-cuda-c"),
                 key=lambda r: r["median"])
    int8 = [r for r in best.values() if r["dtype"] == "int8"]
    top8 = max(int8, key=lambda r: r["median"])
    numpy = best.get(("numpy", "int8"))
    c = bench["ceiling"]
    m = {
        "benchProvisional": "true" if bench.get("provisional") else "false",
        "benchMode": bench.get("mode", "?"),
        "benchFile": bench_path.name.replace("_", r"\_"),
        "gpuName": bench["machine"]["gpu"],
        "copyBW": fmt(c["copy_bandwidth_GBs"]),
        "specBW": fmt(c["spec_bandwidth_GBs"], 0),
        "philoxRate": fmt(c["philox_calls_per_ns"]),
        "callsPerWord": fmt(c["msc_philox_calls_per_word_at_Tc"], 2),
        "bitsPerSpin": fmt(c["msc_philox_calls_per_word_at_Tc"] * 128 / 64, 1),
        "memCeilInt": fmt(c["copy_bandwidth_GBs"] / 3.0, 0),
        "memCeilMsc": fmt(c["copy_bandwidth_GBs"] / 0.375, 0),
        "rngCeilInt": fmt(c["philox_calls_per_ns"] * 4, 0),
        "rngCeilMsc": fmt(c["philox_calls_per_ns"] * 64 / c["msc_philox_calls_per_word_at_Tc"],
                          0),
        "callsPerWordWarp": fmt(c.get("msc_philox_calls_per_word_warp_max_at_Tc", float("nan")),
                                2),
        "bitsPerSpinWarp": fmt(c.get("msc_philox_calls_per_word_warp_max_at_Tc", float("nan"))
                               * 2, 1),
        "rngCeilMscWarp": fmt(c["philox_calls_per_ns"] * 64
                              / c.get("msc_philox_calls_per_word_warp_max_at_Tc", float("nan")),
                              0),
        "bestFlips": fmt(top["median"]),
        "bestBackend": label(top),
        "bestL": str(top["size"][0]),
        "bestPctCopy": fmt(top["pct_copy_bw"]),
        "bestPctSpec": fmt(top["pct_spec_bw"]),
        "bestGBs": fmt(top["effective_GBs"]),
        "bestPyFlips": fmt(top_py["median"]),
        "bestPyBackend": label(top_py),
        "bestPyPctCopy": fmt(top_py["pct_copy_bw"]),
        "bestIntFlips": fmt(top8["median"]),
        "bestIntBackend": label(top8),
        "bestIntPctCopy": fmt(top8["pct_copy_bw"]),
        "numpyFlips": fmt(numpy["median"], 3) if numpy else "n/a",
        "speedupVsNumpy": fmt(top["median"] / numpy["median"], 0) if numpy else "n/a",
        "nRuns": str(len(bench["runs"])),
        "maxSpins": f"{max(r['size'][0] ** r['dim'] for r in bench['runs']):.3g}".replace(
            "e+", r"\times 10^{") + "}",
    }
    for key, tag in ((("cupy-cuda-c", "msc-eager32"), "Eager"),
                     (("cupy-cuda-c", "msc"), "Lazy"),
                     (("cupy-cuda-c", "int8-rng-per-site"), "PerSite"),
                     (("cupy-cuda-c", "int8"), "CudaInt")):
        m[f"abl{tag}"] = fmt(best[key]["median"]) if key in best else "n/a"
    # best Python-DSL kernel as a fraction of the CUDA C reference, per spin format
    for variant, tag in (("int8", "Int"), ("msc", "Msc")):
        dsl = [r for (b, v), r in best.items() if v == variant and b in ("triton", "warp", "numba")]
        ref = best.get(("cupy-cuda-c", variant))
        if dsl and ref:
            top_dsl = max(dsl, key=lambda r: r["median"])
            m[f"dsl{tag}Backend"] = LABEL[top_dsl["backend"]]
            m[f"dsl{tag}PctCuda"] = fmt(100 * top_dsl["median"] / ref["median"], 0)
    m["bestPctRngWarp"] = fmt(100 * top["median"] / (
        c["philox_calls_per_ns"] * 64 / c.get("msc_philox_calls_per_word_warp_max_at_Tc",
                                              float("nan"))))
    if val:
        s = val["summary"]
        m.update({
            "valQuick": "true" if val.get("quick") else "false",
            "valIdentical": "all" if s["all_identical"] else "NOT all",
            "valNIdentity": str(s["n_identity_checks"]),
            "valEmaxErr": f"{s['e_max_abs_err_vs_exact_L']:.1e}".replace("e-0", r"\times 10^{-")
            + "}",
            "valEmaxZ": fmt(s["e_max_abs_z"], 2),
            "valMmaxErr": f"{s['m_max_abs_err_vs_yang_away_from_Tc']:.1e}".replace(
                "e-0", r"\times 10^{-") + "}",
            "valTcTwo": fmt(s["Tc2d_estimate"], 4),
            "valTcTwoErr": fmt(100 * s["Tc2d_rel_error"], 2),
            "valTcThree": fmt(s["Tc3d_estimate"], 4),
            "valTcThreeErr": fmt(100 * s["Tc3d_rel_error"], 2),
            "valThreeZ": fmt(s["e3d_max_abs_z_vs_wolff"], 2),
            "valLtwo": str(val["scan2d"][0]["L"]),
            "valLthree": str(val["scan3d"][0]["L"]),
        })
    lines = ["% GENERATED by figures/make_figures.py from "
             f"results/{bench_path.name} -- do not edit by hand."]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    # performance table
    tab = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"backend & variant & $L$ & flips/ns & GB/s & \% copy \\", r"\midrule"]
    for r in sorted(best.values(), key=lambda r: (r["dtype"] != "int8",
                                                  ORDER.index(r["backend"]), r["variant"])):
        tab.append(f"{LABEL[r['backend']]} & {r['variant']} & {r['size'][0]} & "
                   f"{r['median']:.2f} & {r['effective_GBs']:.1f} & {r['pct_copy_bw']:.1f} \\\\")
    tab += [r"\bottomrule", r"\end{tabular}"]
    lines.append("\\newcommand{\\perfTable}{" + "\n".join(tab) + "}")
    t3 = [r for r in bench["runs"] if r["dim"] == 3]
    b3 = max(t3, key=lambda r: r["median"]) if t3 else None
    for k, v in (("bestThreeFlips", fmt(b3["median"]) if b3 else "n/a"),
                 ("bestThreeBackend", label(b3) if b3 else "n/a"),
                 ("bestThreeL", str(b3["size"][0]) if b3 else "n/a"),
                 ("bestThreePct", fmt(b3["pct_copy_bw"]) if b3 else "n/a")):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (ROOT / "paper" / "results.tex").write_text("\n".join(lines) + "\n")


def readme_table(bench: dict) -> str:
    """Markdown performance table for the README (printed; pasted by the coordinator)."""
    best = best_per_variant(runs_2d_metro(bench))
    out = ["| backend | variant | L | flips/ns | eff. GB/s | % of copy BW |",
           "|---|---|---:|---:|---:|---:|"]
    for r in sorted(best.values(), key=lambda r: (r["dtype"] != "int8",
                                                  ORDER.index(r["backend"]), r["variant"])):
        out.append(f"| {LABEL[r['backend']]} | {r['variant']} | {r['size'][0]} | "
                   f"{r['median']:.2f} | {r['effective_GBs']:.1f} | {r['pct_copy_bw']:.1f} |")
    return "\n".join(out)


def _replace_block(text: str, tag: str, body: str) -> str:
    start, end = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + "\n" + body.strip() + "\n" + text[j:]


def update_readme(bench: dict, val: dict | None, bench_path: Path) -> None:
    """Fill the README's generated blocks (headline, table, validation) from the JSON."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    best = best_per_variant(runs_2d_metro(bench))
    top = max(best.values(), key=lambda r: r["median"])
    top_py = max((r for r in best.values() if r["backend"] != "cupy-cuda-c"),
                 key=lambda r: r["median"])
    c = bench["ceiling"]
    prov = ("\n> **PROVISIONAL**: development run on a shared GPU "
            f"(`results/{bench_path.name}`); final numbers come from `bench/run.py --full`.\n"
            if bench.get("provisional") else "")
    head = (f"{prov}\n| | |\n|---|---|\n"
            f"| **Fastest variant** | {label(top)}, L = {top['size'][0]} |\n"
            f"| **Throughput** | **{top['median']:.1f} spin flips / ns** "
            "(2D Metropolis, T = T_c) |\n"
            f"| **% of ceiling** | {top['pct_copy_bw']:.1f} % of measured copy bandwidth "
            f"({c['copy_bandwidth_GBs']:.0f} GB/s), {top['pct_spec_bw']:.1f} % of nominal "
            f"({c['spec_bandwidth_GBs']:.0f} GB/s), at {top['bytes_per_unit']} B/update |\n"
            f"| **Fastest pure-Python-DSL kernel** | {label(top_py)}: {top_py['median']:.1f} "
            f"flips/ns ({top_py['pct_copy_bw']:.1f} % of copy BW) |\n"
            f"| **GPU** | {bench['machine']['gpu']} (single laptop GPU) |\n")
    text = readme.read_text()
    text = _replace_block(text, "headline", head)
    text = _replace_block(text, "table", f"Source: `results/{bench_path.name}`.\n\n"
                          + readme_table(bench))
    if val:
        s = val["summary"]
        vb = ("| check | result |\n|---|---|\n"
              f"| bitwise identity across backends | {s['n_identity_checks']} checks, "
              f"{'all identical' if s['all_identical'] else 'NOT all identical'} |\n"
              "| 2D energy vs exact finite torus | max abs error "
              f"{s['e_max_abs_err_vs_exact_L']:.1e}, "
              f"max {s['e_max_abs_z']:.2f} sigma |\n"
              f"| 2D abs(m) vs Onsager-Yang (T < T_c - 0.2) | max abs error "
              f"{s['m_max_abs_err_vs_yang_away_from_Tc']:.1e} |\n"
              f"| 2D T_c from Binder crossing | {s['Tc2d_estimate']:.4f} "
              f"({100 * s['Tc2d_rel_error']:.2f} % from Onsager) |\n"
              f"| 3D T_c from Binder crossing | {s['Tc3d_estimate']:.4f} "
              f"({100 * s['Tc3d_rel_error']:.2f} % from Ferrenberg et al. 2018) |\n"
              f"| 3D GPU vs Wolff energies | max {s['e3d_max_abs_z_vs_wolff']:.2f} sigma |\n")
        if val.get("quick"):
            vb = "> quick validation run (small lattices)\n\n" + vb
        text = _replace_block(text, "validation", vb)
    readme.write_text(text)


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None)
    ap.add_argument("--validation", default=str(ROOT / "results" / "validation.json"))
    args = ap.parse_args()
    bp = pick_bench(args.bench)
    bench = json.loads(bp.read_text())
    vp = Path(args.validation)
    if not vp.exists():
        vp = ROOT / "results" / "dev-validation-quick.json"
    val = json.loads(vp.read_text()) if vp.exists() else None
    fig_ladder(bench)
    fig_scaling(bench)
    fig_roofline(bench)
    if val:
        fig_validation(val)
        fig_binder(val)
    write_results_tex(bench, val, bp)
    update_readme(bench, val, bp)
    print(f"figures from {bp.name}" + (f" + {vp.name}" if val else "") +
          f"; provisional={bench.get('provisional')}")


if __name__ == "__main__":
    main()
