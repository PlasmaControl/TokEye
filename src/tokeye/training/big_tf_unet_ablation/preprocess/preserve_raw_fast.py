"""Preserve the fast fluctuation diagnostics for the original training shots.

The raw shot data lives in external dirs (``/scratch/gpfs/EKOLEMEN/...``) that have
been wiped/changed repeatedly. This builds a self-contained project-local copy of all
four fast fluctuation diagnostics the ablation trains on, one file per shot, so the
inputs survive any external change and the loader reads a single file per shot.

ece/mhr/bes are extracted from the ~18 GB MDSplus dumps; co2 is folded in from its
per-shot sidecars (``--co2-dir``, default ``data/autoprocess/co2_topup``). ALL channels
of each diagnostic are kept (not a pre-selected subset) so the loader can choose the
training channels later without re-fetching.

Output layout, one file per shot (``{out_dir}/{shot}.h5``)::

    /ece   (attrs: t0_ms, dt_ms, n, rate_khz, units, source_tree)   # 500 kHz
        TECEF01 (float32, n)  ... [attr: source = full MDSplus node path]
    /mhr   (attrs: ...)   # magnetics high resolution, probes B1..B8 @ 2 MHz (NOT Mirnov/MPI)
        B1 (float32, n) ... B8
    /bes   (attrs: ...)   # beam emission spectroscopy @ 1 MHz; absent for no-beam shots
        BESFU01 (float32, n) ...
    /co2   (attrs: ...)   # BCI interferometer chords, ~2 MHz (folded in from sidecar)
        DENR0UF (float32, n) DENV1UF DENV2UF DENV3UF

Time base is uniform and shared within each diagnostic (verified), so it is stored
as (t0_ms, dt_ms, n) attrs rather than a per-channel float64 dim0 array.

Run::

    python -m tokeye.training.big_tf_unet_ablation.preprocess.preserve_raw_fast
    python -m ...preserve_raw_fast --overwrite --shots-file data/autoprocess/settings/shots.txt
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# (modality group name, tree filter, channel-name regex). The three fast
# fluctuation diagnostics the ablation trains on:
#   ece -- ECE radiometer TECEFnn (uppercase ``D3D`` tree; a duplicate lowercase
#          ``d3d`` tree is excluded by the tree filter), 500 kHz.
#   mhr -- "magnetics high resolution", the 8 probes B1..B8 (PTDATA), 2 MHz.
#          NOT the Mirnov array (MPI*) -- that is a separate diagnostic, unused here.
#   bes -- beam emission spectroscopy BESFUnn (PTDATA), 1 MHz.
MODALITIES = [
    ("ece", "/D3D/", r"^TECEF\d+$"),
    ("mhr", "/PTDATA/", r"^B[1-8]$"),
    ("bes", "/PTDATA/", r"^BESFU\d+$"),
]

DEFAULT_SRC = Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/d3d_time_series_data")
DEFAULT_OUT = Path("data/autoprocess/raw_fast")
DEFAULT_SHOTS = Path("data/autoprocess/settings/shots.txt")
DEFAULT_CO2 = Path("data/autoprocess/co2_topup")  # co2 sidecars (folded in as /co2)


def _chan_name(node: str) -> str:
    """Short channel name: drop the MDSplus tree prefix (after the last ':').

    ``\\D3D::TOP.ELECTRONS.ECE.TECEF:TECEF08`` -> ``TECEF08``; PTDATA names
    (``MPI1A011D``, ``BESFU01``) have no ':' and pass through unchanged.
    """
    return node.split("/")[-2].split(":")[-1]


def _channel_leaves(f: h5py.File, tree: str, pattern: str) -> list[str]:
    """All ``/data`` leaves for one diagnostic, deduped by channel name.

    ``pattern`` is an exact regex on the short channel name (e.g. ``^B[1-8]$``)
    so e.g. the mhr probes B1..B8 are not confused with BESFU/BCOIL/BT.
    """
    rx = re.compile(pattern)
    names: list[str] = []
    f.visit(names.append)
    out: dict[str, str] = {}
    for n in names:
        if not n.endswith("/data") or tree not in n:
            continue
        chan = _chan_name(n)
        if rx.match(chan) and chan not in out:
            out[chan] = n
    return [out[k] for k in sorted(out)]


def _time_base(f: h5py.File, data_node: str) -> dict:
    dim0_node = data_node.rsplit("/", 1)[0] + "/dim0"
    t = f[dim0_node][:].astype(np.float64)
    n = len(t)
    dt = float((t[-1] - t[0]) / (n - 1)) if n > 1 else 0.0
    units_node = data_node.rsplit("/", 1)[0] + "/units"
    units = ""
    if units_node in f:
        u = f[units_node][()]
        units = u.decode() if isinstance(u, bytes) else str(u)
    return {"t0_ms": float(t[0]), "dt_ms": dt, "n": n,
            "rate_khz": (1.0 / dt) if dt else 0.0, "units": units}


def _add_co2(fout: h5py.File, shot: str, co2_dir: Path) -> int:
    """Fold the co2 sidecar (``{co2_dir}/{shot}_co2.h5``) in as a ``/co2`` group.

    co2 is fetched separately (different digitizer) into per-shot sidecars; merging
    it here makes each output file self-contained (all 4 modalities, one file).
    """
    co2_src = co2_dir / f"{shot}_co2.h5"
    if not co2_src.exists():
        return 0
    with h5py.File(co2_src, "r") as fco2:
        names: list[str] = []
        fco2.visit(names.append)
        leaves = sorted(
            n for n in names
            if n.endswith("/data") and isinstance(fco2.get(n), h5py.Dataset)
        )
        if not leaves:
            return 0
        g = fout.create_group("co2")
        for k, v in _time_base(fco2, leaves[0]).items():
            g.attrs[k] = v
        for node in leaves:
            d = g.create_dataset(_chan_name(node), data=fco2[node][:].astype(np.float32))
            d.attrs["source"] = node
    return len(leaves)


def preserve_shot(shot: str, src_dir: Path, out_dir: Path, overwrite: bool,
                  co2_dir: Path | None = None) -> dict:
    src = src_dir / f"{shot}.h5"
    out = out_dir / f"{shot}.h5"
    if not src.exists():
        return {"shot": shot, "status": "src_missing"}
    if out.exists() and not overwrite:
        return {"shot": shot, "status": "skip_exists"}

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    tmp = out.with_suffix(".h5.tmp")
    try:
        with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
            fout.attrs["shot"] = shot
            fout.attrs["source"] = str(src)
            for mod, tree, pattern in MODALITIES:
                leaves = _channel_leaves(fin, tree, pattern)
                counts[mod] = len(leaves)
                if not leaves:
                    continue
                g = fout.create_group(mod)
                for k, v in _time_base(fin, leaves[0]).items():
                    g.attrs[k] = v
                g.attrs["source_tree"] = tree
                for node in leaves:
                    d = g.create_dataset(_chan_name(node), data=fin[node][:].astype(np.float32))
                    d.attrs["source"] = node
            if co2_dir is not None:
                counts["co2"] = _add_co2(fout, shot, co2_dir)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"shot": shot, "status": f"error: {e}"}
    tmp.replace(out)  # atomic: only a complete file appears at the final path
    return {"shot": shot, "status": "ok", "size_gb": out.stat().st_size / 1e9, **counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=Path, default=DEFAULT_SHOTS)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--co2-dir", type=Path, default=DEFAULT_CO2,
                    help="co2 sidecar dir folded in as /co2 (pass '' to skip)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    co2_dir = args.co2_dir if args.co2_dir and str(args.co2_dir) else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shots = args.shots_file.read_text().split()
    logger.info(f"Preserving {len(shots)} shots: {args.src_dir} -> {args.out_dir}")

    results = []
    for i, shot in enumerate(shots, 1):
        r = preserve_shot(shot, args.src_dir, args.out_dir, args.overwrite, co2_dir)
        results.append(r)
        logger.info(f"[{i}/{len(shots)}] {shot}: {r.get('status')} "
                    f"ece={r.get('ece','-')} mhr={r.get('mhr','-')} bes={r.get('bes','-')} "
                    f"co2={r.get('co2','-')} {r.get('size_gb',0):.2f}GB")

    ok = [r for r in results if r["status"] == "ok"]
    total = sum(r.get("size_gb", 0) for r in ok)
    logger.info(f"DONE: {len(ok)}/{len(shots)} preserved, {total:.1f}GB total")
    bad = [r for r in results if r["status"] not in ("ok", "skip_exists")]
    if bad:
        logger.warning(f"problems: {bad}")


if __name__ == "__main__":
    main()
