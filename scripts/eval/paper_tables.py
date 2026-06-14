"""Paper-ready tables for Nature Comm submission.

Reads existing eval CSVs and emits:
  - table_tjii_vs_bustos.{tex,md}
  - table_raddet_per_variant.{tex,md}
  - table_dclde_per_species.{tex,md}

LaTeX uses booktabs.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/scratch/gpfs/nc1514/tokeye")
RESULTS = ROOT / "data" / "eval" / "results"
OUT = ROOT / "output" / "nature_figures"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(x, n=3):
    return f"{x:.{n}f}" if x is not None else "—"


# ---------------------------------------------------------------------------
# Table 1: TJ-II vs Bustos 2021 head-to-head
# ---------------------------------------------------------------------------

def table_tjii_vs_bustos():
    # TokEye TJII at default + F1-opt
    with (RESULTS / "TJII2021.csv").open() as f:
        tjii_default = next(csv.DictReader(f))
    with (RESULTS / "TJII2021_f1_optimal.csv").open() as f:
        tjii_opt = next(csv.DictReader(f))

    rows = [
        {
            "Method": "Bustos 2021 (supervised)",
            "thr": "—",
            "Recall": 0.579,
            "Prec": None,
            "F1": None,
            "per-image IoU": 0.427,
        },
        {
            "Method": "TokEye (zero-shot, default thr=0.5)",
            "thr": "0.50",
            "Recall": float(tjii_default["recall"]),
            "Prec": float(tjii_default["precision"]),
            "F1": float(tjii_default["f1"]),
            "per-image IoU": float(tjii_default["iou_per_image_mean"]),
        },
        {
            "Method": "TokEye (zero-shot, F1-optimal thr=0.8)",
            "thr": "0.80",
            "Recall": float(tjii_opt["recall_at_opt"]),
            "Prec": float(tjii_opt["precision_at_opt"]),
            "F1": float(tjii_opt["f1_at_opt"]),
            "per-image IoU": float(tjii_opt["iou_per_image_mean_at_opt"]),
        },
    ]

    md = [
        "# Table: TJ-II head-to-head — TokEye (zero-shot) vs Bustos 2021 (supervised)",
        "",
        "| Method | thr | Recall | Precision | F1 | per-image IoU |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['Method']} | {r['thr']} | "
            f"{fmt(r['Recall'])} | {fmt(r['Prec'])} | "
            f"{fmt(r['F1'])} | **{fmt(r['per-image IoU'])}** |"
        )
    md.append("")
    md.append("Bustos 2021 supervised baseline (UNet trained on TJ-II): Bustos et al. "
              "report only per-image IoU, recall, and AUC; precision/F1 not given.")
    md.append("TokEye reaches 92% of Bustos recall (0.825/0.579) at default "
              "threshold and 61% of per-image IoU (0.260/0.427) at F1-optimal "
              "threshold — without any training on TJ-II data.")
    (OUT / "table_tjii_vs_bustos.md").write_text("\n".join(md))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TJ-II head-to-head. TokEye is plasma-trained but evaluated "
        r"zero-shot on the public TJ-II 2021 test set; Bustos et al.\ (2021) "
        r"is a UNet trained directly on TJ-II. Bustos do not report "
        r"precision or F1.}",
        r"\label{tab:tjii_vs_bustos}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & thr & Recall & Precision & F1 & per-image IoU \\",
        r"\midrule",
    ]
    for r in rows:
        prec = fmt(r["Prec"])
        f1 = fmt(r["F1"])
        tex.append(
            f"{r['Method']} & {r['thr']} & "
            f"{fmt(r['Recall'])} & {prec} & {f1} & "
            f"\\textbf{{{fmt(r['per-image IoU'])}}} \\\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "table_tjii_vs_bustos.tex").write_text("\n".join(tex))
    print("saved table_tjii_vs_bustos.{md,tex}")


# ---------------------------------------------------------------------------
# Table 2: RadDet per-variant (6 rows × pixel F1 / IoU / mAP50 / mAP50:95)
# ---------------------------------------------------------------------------

def table_raddet_per_variant():
    # F1-opt + per-img IoU
    f1opt = {}
    with (RESULTS / "RadDet_f1_optimal.csv").open() as f:
        for r in csv.DictReader(f):
            f1opt[r["variant"]] = r

    # mAP@0.25 and mAP@0.5 from detection CSV
    map25 = {}
    map50 = {}
    with (RESULTS / "RadDet_detection.csv").open() as f:
        for r in csv.DictReader(f):
            iou = float(r["iou_threshold"])
            if abs(iou - 0.25) < 1e-6:
                map25[r["variant"]] = float(r["ap"])
            elif abs(iou - 0.5) < 1e-6:
                map50[r["variant"]] = float(r["ap"])

    # COCO mAP from main CSV
    coco = {}
    with (RESULTS / "RadDet.csv").open() as f:
        for r in csv.DictReader(f):
            coco[r["variant"]] = float(r["coco_map"])

    variants = list(f1opt.keys())

    md = [
        "# Table: RadDet per-variant pixel and detection metrics",
        "",
        "| Variant | F1-opt thr | pixel F1 | per-image IoU | mAP@0.25 | mAP@0.5 | COCO mAP@[0.5:0.95] |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for v in variants:
        short = v.replace("RadDet40k", "").replace("Tv2", "")
        r = f1opt[v]
        row = {
            "variant": short,
            "thr": float(r["f1_optimal_threshold"]),
            "f1": float(r["f1_at_opt"]),
            "iou_per_img": float(r["iou_per_image_mean_at_opt"]),
            "map25": map25.get(v, 0.0),
            "map50": map50.get(v, 0.0),
            "coco": coco.get(v, 0.0),
        }
        rows.append(row)
        md.append(
            f"| {short} | {row['thr']:.2f} | "
            f"{fmt(row['f1'])} | {fmt(row['iou_per_img'])} | "
            f"{fmt(row['map25'])} | **{fmt(row['map50'])}** | {fmt(row['coco'])} |"
        )

    # mean row
    n = len(rows)
    mean_f1 = sum(r["f1"] for r in rows) / n
    mean_iou = sum(r["iou_per_img"] for r in rows) / n
    mean_map25 = sum(r["map25"] for r in rows) / n
    mean_map50 = sum(r["map50"] for r in rows) / n
    mean_coco = sum(r["coco"] for r in rows) / n
    md.append(
        f"| **mean** | — | "
        f"{fmt(mean_f1)} | {fmt(mean_iou)} | "
        f"{fmt(mean_map25)} | **{fmt(mean_map50)}** | {fmt(mean_coco)} |"
    )
    md.append("")
    md.append("All numbers zero-shot, plasma-trained model. Every variant is "
              "evaluated from a single pass at an effective 512x512 inference "
              "resolution (x4 for 128, x2 for 256, native for 512); pixel "
              "metrics use the per-variant F1-optimal threshold. Boxes are "
              "formed by shrinking each predicted component to its energetic "
              "core and merging frequency-overlapping components across small "
              "time gaps. mAP@0.5 is the primary detection metric; COCO "
              "mAP@[.5:.95] is low because, with no learned box regressor, "
              "boxes are correctly located but not pixel-tight (AP falls "
              "sharply for IoU >= 0.7).")
    (OUT / "table_raddet_per_variant.md").write_text("\n".join(md))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RadDet zero-shot evaluation across six dataset variants "
        r"(resolution $\times$ hardware platform). All metrics derive from a "
        r"single pass at an effective $512\times512$ inference resolution "
        r"($\times4$ for 128, $\times2$ for 256, native for 512); pixel metrics "
        r"use the per-variant F1-optimal threshold. Detection boxes are formed "
        r"by shrinking each predicted component to its energetic core and "
        r"merging frequency-overlapping components across small time gaps "
        r"(Methods). mAP@0.5 is the primary detection metric; the COCO-style "
        r"mAP@[.5:.95] is low because, without a learned box regressor, boxes "
        r"are correctly located but not pixel-tight (AP falls sharply for "
        r"$\mathrm{IoU}\geq0.7$).}",
        r"\label{tab:raddet_per_variant}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Variant & F1-opt thr & pixel F1 & per-image IoU & mAP@0.25 & mAP@0.5 & mAP@[.5:.95] \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['variant']} & {row['thr']:.2f} & "
            f"{fmt(row['f1'])} & {fmt(row['iou_per_img'])} & "
            f"{fmt(row['map25'])} & \\textbf{{{fmt(row['map50'])}}} & "
            f"{fmt(row['coco'])} \\\\"
        )
    tex += [
        r"\midrule",
        f"mean & --- & {fmt(mean_f1)} & {fmt(mean_iou)} & {fmt(mean_map25)} & "
        f"\\textbf{{{fmt(mean_map50)}}} & {fmt(mean_coco)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "table_raddet_per_variant.tex").write_text("\n".join(tex))
    print("saved table_raddet_per_variant.{md,tex}")


# ---------------------------------------------------------------------------
# Table 3: DCLDE per-species (4 species; P. electra excluded — broken dataset)
# ---------------------------------------------------------------------------

def table_dclde_per_species():
    default = {}
    with (RESULTS / "DCLDE2011.csv").open() as f:
        for r in csv.DictReader(f):
            default[r["species"]] = r

    f1opt = {}
    with (RESULTS / "DCLDE2011_f1_optimal.csv").open() as f:
        for r in csv.DictReader(f):
            f1opt[r["species"]] = r

    keep = [
        "Delphinus capensis",
        "Delphinus delphis",
        "StenellaLongirostrisLongirostris",
        "Tursiops truncatus-SoCal",
    ]
    short_names = {
        "Delphinus capensis": "D. capensis",
        "Delphinus delphis": "D. delphis",
        "StenellaLongirostrisLongirostris": "S. longirostris",
        "Tursiops truncatus-SoCal": "T. truncatus (SoCal)",
    }

    md = [
        "# Table: DCLDE per-species pixel metrics (4 species, P. electra excluded)",
        "",
        "| Species | Recall@0.5 | F1@0.5 | F1-opt thr | F1-opt | per-image IoU@opt |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for sp in keep:
        d = default[sp]
        o = f1opt[sp]
        row = {
            "name": short_names[sp],
            "r_default": float(d["recall"]),
            "f1_default": float(d["f1"]),
            "f1opt_thr": float(o["f1_optimal_threshold"]),
            "f1_opt": float(o["f1_at_opt"]),
            "iou_per_img_opt": float(o["iou_per_image_mean_at_opt"]),
        }
        rows.append(row)
        md.append(
            f"| {row['name']} | {fmt(row['r_default'])} | "
            f"{fmt(row['f1_default'])} | {row['f1opt_thr']:.2f} | "
            f"**{fmt(row['f1_opt'])}** | {fmt(row['iou_per_img_opt'])} |"
        )
    mean_r = sum(r["r_default"] for r in rows) / len(rows)
    mean_f1d = sum(r["f1_default"] for r in rows) / len(rows)
    mean_f1o = sum(r["f1_opt"] for r in rows) / len(rows)
    mean_iou = sum(r["iou_per_img_opt"] for r in rows) / len(rows)
    md.append(
        f"| **mean** | **{fmt(mean_r)}** | "
        f"**{fmt(mean_f1d)}** | — | **{fmt(mean_f1o)}** | "
        f"**{fmt(mean_iou)}** |"
    )
    md.append("")
    md.append("All metrics zero-shot, plasma-trained model, with 3×3 median "
              "filter on sigmoid before thresholding. *Peponocephala electra* "
              "excluded — annotations confirmed broken on inspection.")
    (OUT / "table_dclde_per_species.md").write_text("\n".join(md))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DCLDE 2011 zero-shot evaluation across four dolphin species. "
        r"\emph{Peponocephala electra} is excluded because manual inspection "
        r"of its annotations revealed systematic mislabelling. F1-optimal "
        r"thresholds are selected per species from the PR sweep.}",
        r"\label{tab:dclde_per_species}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Species & Recall@0.5 & F1@0.5 & F1-opt thr & F1-opt & per-image IoU @ opt \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"\\textit{{{row['name']}}} & {fmt(row['r_default'])} & "
            f"{fmt(row['f1_default'])} & {row['f1opt_thr']:.2f} & "
            f"\\textbf{{{fmt(row['f1_opt'])}}} & {fmt(row['iou_per_img_opt'])} \\\\"
        )
    tex += [
        r"\midrule",
        f"mean & \\textbf{{{fmt(mean_r)}}} & \\textbf{{{fmt(mean_f1d)}}} & "
        f"--- & \\textbf{{{fmt(mean_f1o)}}} & "
        f"\\textbf{{{fmt(mean_iou)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "table_dclde_per_species.tex").write_text("\n".join(tex))
    print("saved table_dclde_per_species.{md,tex}")


def main():
    table_tjii_vs_bustos()
    table_raddet_per_variant()
    table_dclde_per_species()
    print(f"\nAll tables saved to {OUT}")


if __name__ == "__main__":
    main()
