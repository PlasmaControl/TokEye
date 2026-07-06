"""
Cross-coherence between Mirnov probe coils (MPI) at EPM frequencies (8–20 kHz)
during quiet vs active EPM phases.

MPI coils are PTDATA from the D3D tree.  They sample at ~200 kHz and directly
resolve the 8–20 kHz EPM band without the aliasing that affects ECE (5 kHz).

Key probe pairs:
  Phi = 157° (SET A): MPI66M157D (midplane), MPI3U157D, MPI1U157D (upper)
  Phi = 322° (SET B): MPI11M322D (midplane), MPI3A322D, MPI1A322D (upper)
  Toroidal separation: 165° → phase shift for n=1 should be ±165°

Strategy:
  1. Cross-coherence between same-poloidal pairs at different Phi → detects n=1
  2. Coherence magnitude at 8–20 kHz: quiet vs active
  3. Cross-spectral phase: is it consistent with n=1 in both phases?

Run: /fusion/projects/codes/conda/omega/envs_public/general/bin/python3 mpi_coherence.py
"""
import sys, os, json
sys.path.insert(0, '/home/yasodak/NTM_premptive_control')
sys.path.insert(0, '/home/yasodak')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import coherence, welch, csd

SHOTS = [199606, 199607]
LONG_EVENT = {199606: (2746, 4056), 199607: (3496, 4801)}

# Segment definitions: inter-burst quiet gaps vs active burst periods.
# "Quiet" = low-N1RMS gaps embedded WITHIN the intermittent burst period,
# NOT the pre-onset flat-top (which has different plasma conditions).
#
# 199607 (long event 3496–4801 ms):
#   quiet_1 ~3400ms: gap between pre-event burst (3340ms) and long-event onset
#                    N1RMS5 max = 0.7–2.3 G over 3360–3475ms
#   active: main burst period, N1RMS5 routinely 12–28 G
#   quiet_2 ~4720ms: gap after last major burst at 4700ms
#
# 199606 (long event 2746–4056 ms, bursts continue to ~4900ms):
#   Inter-burst quiet gaps are very short (< 40ms), insufficient for 5kHz ECE.
#   Use the pre-onset quiet (2000–2740ms) for 199606 as no inter-burst gap
#   is long enough to resolve EPM alias frequencies.
SEGS = {
    199606: {
        'quiet':  (2000, 2740),   # pre-onset (no extended inter-burst quiet available)
        'active': (3460, 3620),   # peak burst period — N1RMS5 consistently 15–20 G
    },
    199607: {
        'quiet':  (3360, 3475),   # inter-burst quiet gap ~3400ms (0.7–2.3 G)
        'active': (3800, 4700),   # main burst period — N1RMS5 12–28 G
        'quiet2': (4720, 4860),   # second quiet gap ~4800ms (2.5–2.0 G)
    },
}

# Probe pairs: (phi_A_name, phi_B_name, poloidal_label, phi_sep_deg)
# phi_sep = B_phi - A_phi — used to predict n=1 phase shift
PROBE_PAIRS = [
    ('MPI66M157D',  'MPI11M322D', 'midplane',   165.0),
    ('MPI3U157D',   'MPI3A322D',  'upper-3',    165.0),
    ('MPI1U157D',   'MPI1A322D',  'upper-1',    165.0),
]

# EPM frequency band
EPM_LO  = 8000   # Hz
EPM_HI  = 20000  # Hz
NPERSEG = 4096   # gives ~50 Hz resolution at 200 kHz

CACHE_DIR = '/home/yasodak/exp'


# ── Caching helpers ─────────────────────────────────────────────────────────
def cache_path(shot, signal):
    return os.path.join(CACHE_DIR, str(shot), f'ptdata_{signal}_{shot}.json')


def load_or_fetch(shot, signal):
    """Return (t_ms, y) from cache; fetch from MDSplus if absent."""
    cp = cache_path(shot, signal)
    if os.path.exists(cp):
        with open(cp) as f:
            d = json.load(f)
        t, y = np.array(d['t']), np.array(d['y'])
        if np.nanmedian(t) < 10:
            t = t * 1e3
        return t, y

    # Not cached — fetch via MDSplus
    print(f'  Fetching {signal} for shot {shot} from atlas.gat.com ...')
    try:
        import MDSplus as mds
        conn = mds.Connection('atlas.gat.com')
        conn.openTree('D3D', shot)
        y = np.array(conn.get(f'PTDATA("{signal}", {shot})').data(), dtype=float)
        t = np.array(conn.get(f'DIM_OF(PTDATA("{signal}", {shot}))').data(), dtype=float)
        conn.closeAllTrees()
    except Exception as e:
        print(f'    FAILED: {e}')
        return None, None

    if t.size == 0:
        print(f'    Empty result for {signal}')
        return None, None

    if np.nanmedian(np.abs(t)) < 10:
        t = t * 1e3   # s → ms
    print(f'    Got {len(t)} pts, t={t[0]:.1f}–{t[-1]:.1f} ms, '
          f'fs≈{1000/np.median(np.diff(t)):.0f} Hz')

    os.makedirs(os.path.join(CACHE_DIR, str(shot)), exist_ok=True)
    with open(cp, 'w') as f:
        json.dump({'t': t.tolist(), 'y': y.tolist()}, f)
    print(f'    Cached → {cp}')
    return t, y


def load_n1rms(shot):
    cp = f'{CACHE_DIR}/{shot}/mds_mhd_MHD__TOP_MIRNOV_N1RMS5.json'
    with open(cp) as f:
        d = json.load(f)
    t, y = np.array(d['t']), np.array(d['y'])
    if np.nanmedian(t) < 10:
        t = t * 1e3
    return t, y


def get_segment(t, y, tmin, tmax):
    mk = (t >= tmin) & (t <= tmax) & np.isfinite(y)
    return t[mk], y[mk]


# ── Main loop ────────────────────────────────────────────────────────────────
for shot in SHOTS:
    print(f'\n=== Shot {shot} ===')
    segs = SEGS[shot]

    # Load all probe signals
    probes = {}
    signals_needed = set()
    for pA, pB, _, _ in PROBE_PAIRS:
        signals_needed.add(pA)
        signals_needed.add(pB)

    for sig in sorted(signals_needed):
        t, y = load_or_fetch(shot, sig)
        if t is not None and len(t) > 100:
            probes[sig] = (t, y)
            fs_est = 1000.0 / np.median(np.diff(t))
            print(f'  {sig}: {len(t)} pts, fs≈{fs_est:.0f} Hz')
        else:
            print(f'  {sig}: not available')

    available_pairs = [(pA, pB, lbl, sep) for pA, pB, lbl, sep in PROBE_PAIRS
                       if pA in probes and pB in probes]
    if not available_pairs:
        print('  No probe pairs available — skipping')
        continue

    # Estimate sampling rate from first available probe
    first_sig = list(probes.keys())[0]
    t0, _ = probes[first_sig]
    FS = 1000.0 / np.median(np.diff(t0))   # Hz
    print(f'  Sampling rate: {FS:.0f} Hz')

    # Recalculate NPERSEG to give ~50 Hz resolution
    nperseg = max(256, int(FS / 50))  # points per segment → ~50 Hz resolution

    # Figure: one row per probe pair, 3 columns (PSD overlay, coherence, phase)
    n_pairs = len(available_pairs)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(15, 3.5 * n_pairs),
                              gridspec_kw={'wspace': 0.3, 'hspace': 0.35})
    if n_pairs == 1:
        axes = axes[np.newaxis, :]
    q2_lbl = f'  quiet2={segs["quiet2"]}ms' if 'quiet2' in segs else ''
    fig.suptitle(f'Shot {shot}: MPI cross-coherence at EPM band (8–20 kHz)\n'
                 f'quiet={segs["quiet"]}ms  active={segs["active"]}ms{q2_lbl}', fontsize=11)

    # Build phase list: quiet, active, and optionally quiet2
    phase_list = [
        ('quiet',  segs['quiet'],  'tab:green',  'quiet ~3400ms'),
        ('active', segs['active'], 'tab:red',    'active burst'),
    ]
    if 'quiet2' in segs:
        phase_list.append(('quiet2', segs['quiet2'], 'tab:blue', 'quiet ~4800ms'))

    print(f'\n  EPM band coherence summary ({EPM_LO}–{EPM_HI} Hz):')
    hdr = '  ' + '  '.join([f'{"coh_"+ph[0]:>12}' for ph in phase_list]) + \
          '  ' + '  '.join([f'{"phase_"+ph[0]+"°":>13}' for ph in phase_list])
    print(f'  {"Pair":<25}{hdr}')

    for row, (pA, pB, lbl, phi_sep) in enumerate(available_pairs):
        tA, yA = probes[pA]
        tB, yB = probes[pB]

        ax_p  = axes[row, 0]   # PSD
        ax_c  = axes[row, 1]   # coherence
        ax_ph = axes[row, 2]   # cross-spectral phase

        coh_band = {}
        phase_band = {}

        for (phase_key, (tmin, tmax), color, _) in phase_list:
            _, yA_seg = get_segment(tA, yA, tmin, tmax)
            _, yB_seg = get_segment(tB, yB, tmin, tmax)
            n_pts = min(len(yA_seg), len(yB_seg))
            if n_pts < nperseg * 4:
                print(f'    {lbl} {phase_key}: insufficient data ({n_pts} pts)')
                continue

            yA_s = yA_seg[:n_pts]
            yB_s = yB_seg[:n_pts]

            # PSD of probe A
            f_p, Pxx = welch(yA_s, fs=FS, nperseg=nperseg)
            ax_p.semilogy(f_p, Pxx, color=color, lw=0.8, alpha=0.8,
                          label=f'{phase_key}')
            ax_p.set_xlim(0, 30000)
            ax_p.axvspan(EPM_LO, EPM_HI, color='yellow', alpha=0.15, zorder=0)

            # Cross-coherence
            f_c, coh = coherence(yA_s, yB_s, fs=FS, nperseg=nperseg)
            ax_c.plot(f_c, coh, color=color, lw=0.8, alpha=0.8,
                      label=f'{phase_key}')
            ax_c.axvspan(EPM_LO, EPM_HI, color='yellow', alpha=0.15, zorder=0)
            ax_c.set_xlim(0, 30000)
            ax_c.set_ylim(0, 1)

            # Cross-spectral phase
            f_s, Pxy = csd(yA_s, yB_s, fs=FS, nperseg=nperseg)
            phase_xy = np.angle(Pxy, deg=True)
            # Weight phase by coherence (plot only where coherence > 0.1)
            mask_coh = coh > 0.1
            ax_ph.scatter(f_s[mask_coh], phase_xy[mask_coh], s=1, c=color,
                          alpha=0.4, label=f'{phase_key}')
            ax_ph.axvspan(EPM_LO, EPM_HI, color='yellow', alpha=0.15, zorder=0)
            ax_ph.set_xlim(0, 30000)
            ax_ph.set_ylim(-180, 180)
            ax_ph.axhline(phi_sep, color='k', lw=0.7, ls='--',
                          label=f'n=1 pred +{phi_sep:.0f}°')
            ax_ph.axhline(phi_sep - 360, color='k', lw=0.7, ls=':')

            # Mean coherence in EPM band
            epm_mask = (f_c >= EPM_LO) & (f_c <= EPM_HI)
            coh_band[phase_key] = float(np.mean(coh[epm_mask])) if epm_mask.sum() > 0 else np.nan
            # Median phase in EPM band (weighted by coherence²)
            if epm_mask.sum() > 0:
                w = coh[epm_mask] ** 2
                ph_vals = phase_xy[epm_mask]
                if w.sum() > 0:
                    phase_band[phase_key] = float(np.average(ph_vals, weights=w))
                else:
                    phase_band[phase_key] = np.nan
            else:
                phase_band[phase_key] = np.nan

        # Axis labels and titles
        ax_p.set_title(f'{pA} ({lbl})', fontsize=8)
        ax_p.set_xlabel('f (Hz)', fontsize=7)
        ax_p.set_ylabel('PSD (arb²/Hz)', fontsize=7)
        ax_p.legend(fontsize=7)
        ax_p.tick_params(labelsize=6)

        ax_c.set_title(f'{pA} × {pB}', fontsize=8)
        ax_c.set_xlabel('f (Hz)', fontsize=7)
        ax_c.set_ylabel('Coherence', fontsize=7)
        ax_c.legend(fontsize=7)
        ax_c.tick_params(labelsize=6)
        ax_c.axhline(2 / nperseg * np.log(20), color='gray', lw=0.7, ls='--',
                     label='95% sig.')

        ax_ph.set_title(f'Phase {pA}→{pB} (Δφ={phi_sep:.0f}°)', fontsize=8)
        ax_ph.set_xlabel('f (Hz)', fontsize=7)
        ax_ph.set_ylabel('Cross-spectral phase (°)', fontsize=7)
        ax_ph.legend(fontsize=6, markerscale=5)
        ax_ph.tick_params(labelsize=6)

        # Print numerical summary for all phases
        coh_vals  = '  '.join([f'{coh_band.get(ph[0], float("nan")):>12.3f}'
                                if not np.isnan(coh_band.get(ph[0], float("nan")))
                                else f'{"---":>12}' for ph in phase_list])
        phase_vals = '  '.join([f'{phase_band.get(ph[0], float("nan")):>13.1f}'
                                 if not np.isnan(phase_band.get(ph[0], float("nan")))
                                 else f'{"---":>13}' for ph in phase_list])
        print(f'  {f"{pA}×{pB}":<25}  {coh_vals}  {phase_vals}  (n=1 pred: ±{phi_sep:.0f}°)')

    out = f'figures/mpi_coherence_{shot}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  Saved {out}')
    plt.close(fig)

print('\nDone.')
