"""Turn outputs/results/*.json + outputs/optuna/*.json into LaTeX tables and copy figures for the report.

    python -m genai.report_assets            # writes report/generated/*.tex and report/figures/*.png
Missing results produce a visible "pending" placeholder, never invented numbers.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import CLASS_TITLES, CLASSES, OUT_DIR, ROOT, load_json

RES, OPT, FIG = OUT_DIR / "results", OUT_DIR / "optuna", OUT_DIR / "figures"


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def table(header, rows, caption, label, colspec=None, small=True) -> str:
    colspec = colspec or ("l" + "r" * (len(header) - 1))
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    size = "\\footnotesize" if small else ""
    head = " & ".join(header)
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{size}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
            f"{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def pending(name: str) -> str:
    return f"\\textit{{\\textcolor{{red}}{{[{esc(name)}: run the experiments, then \\texttt{{python -m genai.report\\_assets}}]}}}}\n"


def f2(x, d=2):
    return f"{x:.{d}f}"


def build(report_dir: Path):
    gen = report_dir / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)

    def write(name, text):
        (gen / f"{name}.tex").write_text(text)

    def need(path):
        return Path(path).exists()

    # ---------------------------------------------------------------- optuna
    rows = []
    for st, nm in (("task1_udae", "Task 1 UDAE"), ("task2_classifier", "Task 2 classifier"), ("task2_specialists", "Task 2 specialists"),
                   ("task3_moe", "Task 3 soft MoE"), ("task4_cgan", "Task 4 cGAN")):
        p = OPT / f"{st}_summary.json"
        if not need(p):
            continue
        s = load_json(p)
        stt = s["trial_states"]
        best = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in s["best_params"].items())
        rows.append([nm, s["n_trials"], stt.get("COMPLETE", 0), stt.get("PRUNED", 0), f2(s["best_value"], 4), esc(best)])
        # per-study search-space table
        sp = table(["Hyper-parameter", "Search space"], [[esc(k), esc(v)] for k, v in s["search_space"].items()],
                   f"Optuna search space for {nm} ({s['n_trials']} trials, {s.get('trial_epochs', '?')} epochs per trial; objective: {esc(s.get('objective', ''))}).",
                   f"tab:space_{st}", "ll")
        write(f"space_{st}", sp)
    write("optuna_summary", table(["Study", "Trials", "Done", "Pruned", "Best obj.", "Selected configuration"], rows,
                                  "Optuna studies: trial counts, best objective value and the final selected configuration.",
                                  "tab:optuna", "lrrrrp{6.2cm}") if rows else pending("Optuna summary"))
    for st in ("task1_udae", "task2_classifier", "task2_specialists", "task3_moe", "task4_cgan"):
        if not need(gen / f"space_{st}.tex"):
            write(f"space_{st}", pending(f"search space {st}"))

    # ---------------------------------------------------------------- task 1
    if need(RES / "task1_test.json"):
        r = load_json(RES / "task1_test.json")
        rows = [[CLASS_TITLES[c], f2(r["by_type"][c]["in_psnr"]), f2(r["by_type"][c]["psnr"]), f2(r["by_type"][c]["in_ssim"], 3),
                 f2(r["by_type"][c]["ssim"], 3), f2(r["by_type"][c]["l1"], 4)] for c in CLASSES]
        write("t1_by_type", table(["Condition", "PSNR in", "PSNR out", "SSIM in", "SSIM out", "L1 out"], rows,
                                  "Task 1 test results by input condition (``in'' = corrupted input vs.\\ clean target, ``out'' = UDAE output).", "tab:t1_type"))
        rows = []
        for c in CLASSES[1:]:
            for lv in ("low", "medium", "high"):
                d = r["by_level"][f"{c}/{lv}"]
                rows.append([CLASS_TITLES[c], lv, f2(d["in_psnr"]), f2(d["psnr"]), f2(d["in_ssim"], 3), f2(d["ssim"], 3)])
        write("t1_by_level", table(["Corruption", "Severity", "PSNR in", "PSNR out", "SSIM in", "SSIM out"], rows,
                                   "Task 1 test results by corruption and fixed severity level.", "tab:t1_level"))
    else:
        write("t1_by_type", pending("Task 1 table")), write("t1_by_level", pending("Task 1 severity table"))

    # ---------------------------------------------------------------- task 2
    if need(RES / "task2_test.json"):
        r = load_json(RES / "task2_test.json")
        cr = r["classifier_test"]
        rows = [[CLASS_TITLES[c], f2(d["precision"], 3), f2(d["recall"], 3), f2(d["f1"], 3), d["support"]] for c, d in cr["per_class"].items()]
        rows.append([r"\textbf{Macro / overall acc.}", f2(cr["macro_precision"], 3), f2(cr["macro_recall"], 3), f2(cr["macro_f1"], 3),
                     f"acc {cr['accuracy']:.3f}"])
        write("t2_classifier", table(["Class", "Precision", "Recall", "F1", "Support"], rows, "Task 2 classifier test metrics.", "tab:t2_cls"))
        rows = []
        for c in CLASSES:
            o, p = r["routing_oracle"]["by_type"][c], r["routing_pred"]["by_type"][c]
            rows.append([CLASS_TITLES[c], f2(o["in_psnr"]), f2(o["psnr"]), f2(p["psnr"]), f2(o["ssim"], 3), f2(p["ssim"], 3)])
        m = r["misrouted"]
        write("t2_routing", table(["Condition", "PSNR in", "PSNR oracle", "PSNR pred.", "SSIM oracle", "SSIM pred."], rows,
                                  f"Task 2 hard routing: oracle vs.\\ predicted routing on the test set. {m['count']} entries "
                                  f"({100 * m['fraction']:.1f}\\%) were misrouted; the mean PSNR loss on those was {m['mean_psnr_drop_when_misrouted']:.2f} dB.",
                                  "tab:t2_route"))
    else:
        write("t2_classifier", pending("Task 2 classifier table")), write("t2_routing", pending("Task 2 routing table"))

    # ---------------------------------------------------------------- task 3
    if need(RES / "task3_test.json"):
        r = load_json(RES / "task3_test.json")
        rows = [[CLASS_TITLES[c], f2(r["by_type"][c]["in_psnr"]), f2(r["by_type"][c]["psnr"]), f2(r["by_type"][c]["ssim"], 3), f2(r["by_type"][c]["l1"], 4)] for c in CLASSES]
        write("t3_by_type", table(["Condition", "PSNR in", "PSNR out", "SSIM out", "L1 out"], rows, "Task 3 soft mixture-of-experts test results.", "tab:t3_type"))
        W = r["routing_weights_by_condition"]
        rows = [[esc(k)] + [f2(v, 2) for v in ws] for k, ws in W.items()]
        write("t3_weights", table(["Condition", "$w_0$ id", "$w_1$ salt", "$w_2$ blur", "$w_3$ occl."], rows,
                                  "Average gate weights per true corruption type and severity (test set).", "tab:t3_w"))
        h = r["routing_health"]
        rows = [[h["branch_names"][k], f2(h["mean_weight_per_branch"][k], 3), f2(h["frac_images_branch_dominant_gt_0.5"][k], 3),
                 f2(h["mean_weight_on_other_types"][h["branch_names"][k]], 3)] for k in range(4)]
        write("t3_health", table(["Branch", "Mean weight", "Frac. $w>0.5$", "Mean $w$ on other types"], rows,
                                 "Routing health: no branch is inactive when its mean weight and dominance fraction are non-zero.", "tab:t3_health"))
        m = r["mixed_corruption_test"]
        rows = [[n.replace("_", r"\_"), f2(m[f"{k}_psnr"]), f2(m[f"{k}_ssim"], 3)] for k, n in
                (("input", "no restoration"), ("universal_ae", "Task 1 universal AE"), ("hard_routed_pred", "Task 2 hard routing (predicted)"), ("soft_moe", "Task 3 soft MoE"))]
        write("t3_mixed", table(["System", "PSNR", "SSIM"], rows, "Two simultaneous corruptions (blur+salt-and-pepper, occlusion+blur).", "tab:t3_mixed", "lrr"))
        if need(RES / "task2_test.json") and need(RES / "task1_test.json"):
            t1, t2 = load_json(RES / "task1_test.json"), load_json(RES / "task2_test.json")
            rows = [[CLASS_TITLES[c], f2(t1["by_type"][c]["psnr"]), f2(t2["routing_oracle"]["by_type"][c]["psnr"]),
                     f2(t2["routing_pred"]["by_type"][c]["psnr"]), f2(r["by_type"][c]["psnr"])] for c in CLASSES]
            rows.append([r"\textbf{All corrupted}", f2(t1["all_corrupted"]["psnr"]), f2(t2["routing_oracle"]["all_corrupted"]["psnr"]),
                         f2(t2["routing_pred"]["all_corrupted"]["psnr"]), f2(r["all_corrupted"]["psnr"])])
            write("compare", table(["PSNR (dB)", "T1 universal", "T2 oracle", "T2 predicted", "T3 soft MoE"], rows,
                                   "PSNR of the four restoration systems on the same test manifest.", "tab:compare"))
        else:
            write("compare", pending("comparison table"))
    else:
        for n in ("t3_by_type", "t3_weights", "t3_health", "t3_mixed", "compare"):
            write(n, pending(n))

    # ---------------------------------------------------------------- task 4
    if need(RES / "task4_test.json"):
        r = load_json(RES / "task4_test.json")
        rows = [[k, f2(v["psnr"]), f2(v["ssim"], 3), f2(v["l1"], 4), v["n"]] for k, v in r["per_style"].items()]
        o, b = r["overall"], r["baseline_grayscale_photo"]
        rows += [[r"\textbf{All styles (cGAN)}", f2(o["psnr"]), f2(o["ssim"], 3), f2(o["l1"], 4), r["n_test"]],
                 ["Grayscale-photo baseline", f2(b["psnr"]), f2(b["ssim"], 3), f2(b["l1"], 4), r["n_test"]]]
        c = r["style_conditioning_check"]
        write("t4_results", table(["Set", "PSNR", "SSIM", "L1", "N"], rows,
                                  f"Task 4 test results against the paired ground-truth sketch. Style-conditioning check: L1 with the correct style {c['l1_matched_style']:.4f} vs.\\ "
                                  f"{c['l1_mismatched_style']:.4f} with a wrong style.", "tab:t4"))
    else:
        write("t4_results", pending("Task 4 table"))

    # ---------------------------------------------------------------- onnx
    if need(OUT_DIR / "onnx_parity.json"):
        rows = [[esc(d["model"]), esc(d.get("file", "")), f"{d['max_abs_diff']:.1e}", d.get("size_mb", ""), "yes" if d["ok"] else "NO"]
                for d in load_json(OUT_DIR / "onnx_parity.json")]
        write("onnx", table(["Model", "File", "Max abs. diff", "MB", "Match"], rows, "ONNX export check: maximum absolute difference between PyTorch and ONNX Runtime outputs.", "tab:onnx", "llrrr"))
    else:
        write("onnx", pending("ONNX parity table"))

    # ---------------------------------------------------------------- dataset sizes
    pets = ROOT / "data" / "pets" / "splits.json"
    if pets.exists():
        s = load_json(pets)
        write("dataset_sizes", f"{len(s['train'])} training, {len(s['val'])} validation")
    else:
        write("dataset_sizes", pending("dataset sizes"))

    # ---------------------------------------------------------------- figures
    n = 0
    if FIG.exists():
        for p in FIG.glob("*.png"):
            shutil.copy(p, report_dir / "figures" / p.name)
            n += 1
    for p in OPT.glob("*.png") if OPT.exists() else []:
        shutil.copy(p, report_dir / "figures" / p.name)
        n += 1
    print(f"report assets written to {report_dir} ({n} figures copied)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default=str(ROOT / "report"))
    build(Path(ap.parse_args().report_dir))


if __name__ == "__main__":
    main()
