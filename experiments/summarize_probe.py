"""Turn `probe_token_sensitivity.py` JSON dumps into a rate-distortion summary.

Prints a table sorted by bitrate, flags which entries are real codecs (a bits/token
figure) versus diagnostic probes, and reports the penalty that structured quantisation
error carries over isotropic noise of the same magnitude.

Usage:
    python experiments/summarize_probe.py outputs/probe_*.json
    python experiments/summarize_probe.py outputs/probe_scene.json --plot outputs/rd.png
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

# Tokens per scene stay fixed across a run, so byte totals are derived from the dump.
FAMILY_PATTERNS = [
    ("noise", re.compile(r"^noise@")),
    ("pca", re.compile(r"^pca@")),
    ("randproj", re.compile(r"^randproj@")),
    ("vq", re.compile(r"^vq@")),
    ("pvq", re.compile(r"^pvq@")),
    ("rvq", re.compile(r"^rvq@")),
    ("uniform", re.compile(r"^uniform@")),
    ("identity", re.compile(r"^identity$")),
]


def family_of(name: str) -> str:
    for label, pattern in FAMILY_PATTERNS:
        if pattern.match(name):
            return label
    return "other"


def series_key(name: str) -> str:
    """Group for plotting: multi-stage codecs are split by codebook size.

    `rvq@8x1024` and `rvq@8x256` sweep different codebooks, so plotting them as one
    curve against bitrate produces a meaningless zigzag.
    """
    match = re.match(r"^(rvq|pvq)@\d+x(\d+)$", name)
    if match:
        return f"{match.group(1)} k={match.group(2)}"
    return family_of(name)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def load(paths: List[str]) -> List[Dict]:
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        data["path"] = path
        runs.append(data)
    return runs


def report_run(run: Dict, top: Optional[int]) -> None:
    tokens = run["token_shape"][0] * run["token_shape"][1]
    print(f"\n{'=' * 92}")
    print(
        f"{run['path']}   target={run['target']}   views={run['views']}   "
        f"tokens={tokens}   gaussians={run['num_gaussians']}"
    )
    print("=" * 92)

    codecs = [r for r in run["results"] if not math.isnan(r["bits_per_token"])]
    probes = [r for r in run["results"] if math.isnan(r["bits_per_token"])]

    if codecs:
        codecs.sort(key=lambda r: r["bits_per_token"])
        print(f"\n  {'codec':<26}{'bits/tok':>9}{'scene':>10}{'rel err':>9}{'PSNR':>8}{'SSIM':>8}{'drift':>8}")
        print("  " + "-" * 88)
        for r in codecs[: top or len(codecs)]:
            total = r["bits_per_token"] / 8 * tokens
            print(
                f"  {r['name']:<26}{r['bits_per_token']:>9.0f}{human_bytes(total):>10}"
                f"{r['token_rel_err']:>9.3f}{r['psnr']:>8.2f}{r['ssim']:>8.3f}"
                f"{r['mean_shift_pct']:>7.2f}%"
            )

    if probes:
        print(f"\n  {'diagnostic':<26}{'rel err':>9}{'PSNR':>8}{'SSIM':>8}{'drift':>8}")
        print("  " + "-" * 61)
        for r in sorted(probes, key=lambda r: r["token_rel_err"]):
            print(
                f"  {r['name']:<26}{r['token_rel_err']:>9.3f}{r['psnr']:>8.2f}"
                f"{r['ssim']:>8.3f}{r['mean_shift_pct']:>7.2f}%"
            )


def report_structure_penalty(runs: List[Dict], tolerance: float = 0.15) -> None:
    """Match each structured entry to the isotropic-noise entry at the same error norm."""
    noise = [
        r
        for run in runs
        for r in run["results"]
        if family_of(r["name"]) == "noise" and r["token_rel_err"] > 0
    ]
    structured = [
        r
        for run in runs
        for r in run["results"]
        if family_of(r["name"]) in {"pca", "vq", "pvq", "rvq", "uniform"}
    ]
    if not noise or not structured:
        return

    rows = []
    for entry in structured:
        err = entry["token_rel_err"]
        match = min(noise, key=lambda n: abs(math.log(n["token_rel_err"] / max(err, 1e-9))))
        if abs(match["token_rel_err"] / max(err, 1e-9) - 1) > tolerance:
            continue
        rows.append((entry, match, match["psnr"] - entry["psnr"]))

    if not rows:
        print("\nNo isotropic-noise control within tolerance; rerun with matched --snr levels.")
        return

    rows.sort(key=lambda t: t[0]["token_rel_err"])
    print(f"\n{'=' * 92}")
    print("Structured-error penalty vs isotropic noise at matched error norm")
    print("=" * 92)
    print(f"\n  {'structured':<26}{'rel err':>9}{'PSNR':>8}{'noise ctrl':>16}{'PSNR':>8}{'penalty':>10}")
    print("  " + "-" * 79)
    for entry, match, penalty in rows:
        print(
            f"  {entry['name']:<26}{entry['token_rel_err']:>9.3f}{entry['psnr']:>8.2f}"
            f"{match['name']:>16}{match['psnr']:>8.2f}{penalty:>9.2f} dB"
        )
    mean_penalty = sum(p for _, _, p in rows) / len(rows)
    print(f"\n  mean penalty: {mean_penalty:.2f} dB over {len(rows)} matched pairs")
    print("  A large penalty means feature-MSE is a poor proxy for downstream quality.")


def make_plot(runs: List[Dict], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_rd, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

    for run in runs:
        label_base = Path(run["path"]).stem.replace("probe_", "")
        by_family: Dict[str, List[Dict]] = {}
        for r in run["results"]:
            by_family.setdefault(series_key(r["name"]), []).append(r)

        for fam, entries in sorted(by_family.items()):
            codecs = sorted(
                (e for e in entries if not math.isnan(e["bits_per_token"])),
                key=lambda e: e["bits_per_token"],
            )
            if len(codecs) > 1:
                ax_rd.plot(
                    [e["bits_per_token"] for e in codecs],
                    [e["psnr"] for e in codecs],
                    marker="o",
                    ms=4,
                    label=f"{label_base}:{fam}",
                )
            usable = sorted(
                (e for e in entries if e["token_rel_err"] > 0 and fam != "identity"),
                key=lambda e: e["token_rel_err"],
            )
            if len(usable) > 1:
                ax_err.plot(
                    [e["token_rel_err"] * 100 for e in usable],
                    [e["psnr"] for e in usable],
                    marker="o",
                    ms=4,
                    label=f"{label_base}:{fam}",
                )

    for ax, xlabel, title in (
        (ax_rd, "Bits per token", "Rate-distortion"),
        (ax_err, "Relative token error (%)", "Distortion vs error norm"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Render PSNR vs unperturbed baseline (dB)")
        ax.set_title(title)
        ax.axhline(30, ls="--", lw=1, color="gray")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsons", nargs="+", help="Probe result files.")
    ap.add_argument("--top", type=int, default=None, help="Show only the N cheapest codecs.")
    ap.add_argument("--plot", default=None, help="Write a rate-distortion figure here.")
    ap.add_argument("--no-penalty", action="store_true", help="Skip the structured-error table.")
    args = ap.parse_args()

    runs = load(args.jsons)
    for run in runs:
        report_run(run, args.top)
    if not args.no_penalty:
        report_structure_penalty(runs)
    if args.plot:
        make_plot(runs, args.plot)


if __name__ == "__main__":
    main()
