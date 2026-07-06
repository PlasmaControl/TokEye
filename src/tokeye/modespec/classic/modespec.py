"""
modespec.py — Python implementation of DIII-D Mirnov mode analysis.

Replicates the core algorithm from /fusion/usc/src/idl/modespec_auto/modespec.pro.
Fetches toroidal Mirnov array (Bp_probes_R0) or poloidal array (Bp_probes_322)
from MDSplus and computes:

  - Power spectrogram (FFT in sliding windows)
  - Toroidal mode number spectrogram (matched-filter fit)
  - RMS amplitude per mode number vs time
  - Single time-slice coherence-weighted phase fit
  - Multi-harmonic least-squares fit (IDL slice_fit equivalent)

Probe geometry from:
  /fusion/projects/diagnostics/magnetics/data/coords/all_mag

Signal naming convention:
  B-dot signals (200 kHz): MPI66M020D, MPI66M067D, ...  (integrated=False)
  Integrated B signals (~20 kHz): MPI66M020, MPI66M067, ... (integrated=True)

Usage (notebook)::

    from modespec import fetch_mirnov, mode_spectrogram, plot_modespec

    signals, t_ms, phi_tor, names = fetch_mirnov(shot, integrated=True)
    result = mode_spectrogram(signals, t_ms, phi_tor,
                              dt_window_ms=10.0, f_smooth_khz=1.0,
                              f_min_khz=1, f_max_khz=8, n_range=(-5, 5))
    fig = plot_modespec(result, shot, n1rms_dict=n1rms[shot])
"""

import numpy as np
import matplotlib.pyplot as plt

# numpy 2.x removed np.trapz (renamed to np.trapezoid); restore the old name so
# this module works under both numpy 1.x (cluster) and 2.x (pixi/conda-forge).
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ── Probe geometry ─────────────────────────────────────────────────────────────
# Bp_probes_R0: midplane toroidal array, 14 probes
# (name, phi_deg) from all_mag. Re-centred: probes > 315° get phi - 360°.
# B-dot names end in D; integrated names are the same without the trailing D.
TOR_PROBES = [
    ('MPI66M020D', 19.5),
    ('MPI66M067D', 67.5),
    ('MPI66M097D', 97.4),
    ('MPI66M127D', 127.9),
    ('MPI66M132D', 132.5),
    ('MPI66M137D', 137.4),
    ('MPI66M157D', 157.6),
    ('MPI66M200D', 199.7),
    ('MPI66M247D', 246.4),
    ('MPI66M277D', 277.5),
    ('MPI66M307D', 307.0),
    ('MPI66M312D', 312.4),
    ('MPI66M322D', 317.4),   # actual location per all_mag, not 322°
    ('MPI66M340D', 339.7),
]
TOR_PROBE_NAMES = [p[0] for p in TOR_PROBES]
TOR_PHI_RAW     = np.array([p[1] for p in TOR_PROBES])
TOR_PHI_DEG     = np.where(TOR_PHI_RAW > 315, TOR_PHI_RAW - 360, TOR_PHI_RAW)
_sort_idx       = np.argsort(TOR_PHI_DEG)
TOR_PROBE_NAMES = [TOR_PROBE_NAMES[i] for i in _sort_idx]
TOR_PHI_DEG     = TOR_PHI_DEG[_sort_idx]

# Bp_probes_322: poloidal array at phi ≈ 322°, 31 probes
# (name, R[m], Z[m]); theta = atan2(Z, R - R0)
POL_PROBES_RAW = [
    ('MPI11M322D', 0.973, -0.002),
    ('MPI1A322D',  0.974,  0.182),
    ('MPI2A322D',  0.974,  0.512),
    ('MPI3A322D',  0.975,  0.850),
    ('MPI4A322D',  0.972,  1.161),
    ('MPI5A322D',  1.051,  1.330),
    ('MPI8A322D',  1.219,  1.406),
    ('MPI89A322D', 1.402,  1.407),
    ('MPI9A322D',  1.584,  1.408),
    ('MPI79FA322D',1.783,  1.323),
    ('MPI79NA322D',1.924,  1.206),
    ('MPI7FA322D', 2.067,  1.090),
    ('MPI7NA322D', 2.219,  0.870),
    ('MPI67A322D', 2.270,  0.746),
    ('MPI6FA322D', 2.319,  0.623),
    ('MPI6NA322D', 2.416,  0.249),
    ('MPI66M322D', 2.418, -0.001),
    ('MPI1B322D',  0.974, -0.187),
    ('MPI2B322D',  0.975, -0.512),
    ('MPI3B322D',  0.974, -0.854),
    ('MPI4B322D',  0.972, -1.159),
    ('MPI5B322D',  1.048, -1.330),
    ('MPI8B322D',  1.254, -1.405),
    ('MPI89B322D', 1.477, -1.406),
    ('MPI9B322D',  1.699, -1.406),
    ('MPI79B322D', 1.894, -1.333),
    ('MPI7FB322D', 2.085, -1.102),
    ('MPI7NB322D', 2.212, -0.873),
    ('MPI67B322D', 2.263, -0.749),
    ('MPI6FB322D', 2.315, -0.624),
    ('MPI6NB322D', 2.416, -0.244),
]
POL_R0 = 1.69
POL_PROBE_NAMES = [p[0] for p in POL_PROBES_RAW]
_R  = np.array([p[1] for p in POL_PROBES_RAW])
_Z  = np.array([p[2] for p in POL_PROBES_RAW])
POL_THETA_DEG = np.degrees(np.arctan2(_Z, _R - POL_R0))
POL_THETA_DEG = np.where(POL_THETA_DEG < -5, POL_THETA_DEG + 360, POL_THETA_DEG)
_psort = np.argsort(POL_THETA_DEG)
POL_PROBE_NAMES = [POL_PROBE_NAMES[i] for i in _psort]
POL_THETA_DEG   = POL_THETA_DEG[_psort]


# ── ECE radial positions ───────────────────────────────────────────────────────

def _ece_freq_ghz(shot, n_ch=40):
    """ECE channel frequencies [GHz] for DIII-D (from getecefreq.py formula)."""
    freqs = np.zeros(n_ch)
    if shot > 178000:
        freqs[:16]  = 82.5  + np.arange(16)          # 82.5–97.5 GHz
        freqs[16:32] = 98.5  + np.arange(16)          # 98.5–113.5 GHz
        freqs[32:40] = 115.5 + 2 * np.arange(8)       # 115.5–129.5 GHz (step 2)
    elif shot > 100600:
        freqs[:16]  = 83.5  + np.arange(16)
        freqs[16:32] = 98.5  + np.arange(16)
        freqs[32:40] = 115.5 + 2 * np.arange(8)
    else:
        freqs[:16]  = 83.5  + np.arange(16)
        freqs[16:32] = 98.5  + np.arange(16)
    return freqs


def ece_channel_radius(shot, bt0_T=None, r_axis_m=None):
    """
    Compute major radius R [m] for each fixed ECE channel.

    Uses second-harmonic emission: f_ECE = 2 × f_ce = 56.0 × B_T [GHz/T].
    With B_T(R) = B_T0 × R_axis / R:  R = 56.0 × B_T0 × R_axis / f_ECE.

    Parameters
    ----------
    shot     : DIII-D shot number (for frequency table lookup)
    bt0_T    : toroidal field at magnetic axis [T].  Default: 1.76 T (typical H-mode)
    r_axis_m : major radius of magnetic axis [m].   Default: 1.69 m

    Returns
    -------
    freq_ghz : (n_ch,)  ECE frequency per channel [GHz]
    R_ece    : (n_ch,)  corresponding major radius [m]
    """
    if bt0_T is None:
        bt0_T    = 1.76
    if r_axis_m is None:
        r_axis_m = 1.69
    freq_ghz = _ece_freq_ghz(shot)
    R_ece    = 56.0 * bt0_T * r_axis_m / freq_ghz
    return freq_ghz, R_ece


# ── MDSplus fetch ──────────────────────────────────────────────────────────────

def fetch_mirnov(shot, array='toroidal', integrated=False,
                 atlas='atlas.gat.com', t_min_ms=None, t_max_ms=None):
    """
    Fetch Mirnov array signals from MDSplus.

    Parameters
    ----------
    shot       : int
    array      : 'toroidal'  → 14-probe midplane toroidal array
                 'poloidal'  → 31-probe 322° poloidal array
    integrated : bool
        False (default) → B-dot signals (names end in D), 200 kHz
        True            → integrated B signals (no trailing D), ~20 kHz
    atlas      : MDSplus server
    t_min_ms, t_max_ms : optional time crop in ms

    Returns
    -------
    signals : ndarray (n_probes, n_t)   [G/s for B-dot; G for integrated]
    t_ms    : ndarray (n_t,)            time axis [ms]
    angles  : ndarray (n_probes,)       phi_tor or theta_pol [deg]
    names   : list of str               PTDATA signal names used
    """
    try:
        import MDSplus as mds
    except ImportError:
        raise RuntimeError(
            'MDSplus is not installed. It ships on the GA cluster and on '
            'conda-forge (conda install -c conda-forge mdsplus); fetching '
            'DIII-D data also requires network access to atlas.gat.com.'
        )

    if array == 'toroidal':
        probe_names = list(TOR_PROBE_NAMES)
        angles      = TOR_PHI_DEG.copy()
    else:
        probe_names = list(POL_PROBE_NAMES)
        angles      = POL_THETA_DEG.copy()

    if integrated:
        # Strip trailing 'D' for integrated B signals
        probe_names = [n[:-1] if n.endswith('D') else n for n in probe_names]

    conn = mds.Connection(atlas)
    conn.openTree('D3D', shot)

    signals    = []
    good_names = []
    good_angles= []
    t_ref      = None   # full time axis from first successful probe

    for name, phi in zip(probe_names, angles):
        bdot_fallback = False
        try:
            d = np.array(conn.get(f'PTDATA("{name}",{shot})').data(), dtype=float)
            t = np.array(conn.get(f'DIM_OF(PTDATA("{name}",{shot}))').data(), dtype=float)
        except Exception:
            if integrated and not name.endswith('D'):
                # Integrated version absent — fall back to B-dot and integrate
                try:
                    bdot_name = name + 'D'
                    d = np.array(conn.get(f'PTDATA("{bdot_name}",{shot})').data(), dtype=float)
                    t = np.array(conn.get(f'DIM_OF(PTDATA("{bdot_name}",{shot}))').data(), dtype=float)
                    bdot_fallback = True
                except Exception as e2:
                    print(f'  Warning: {name} (and {name}D) failed: {e2}')
                    continue
            else:
                import sys
                print(f'  Warning: {name} failed', file=sys.stderr)
                continue

        if t.size < 2:
            continue

        if bdot_fallback:
            # Numerical integration of B-dot → B (cumulative trapezoid)
            dt_s = np.mean(np.diff(t)) * 1e-3   # ms → s
            d = np.cumsum(d) * dt_s              # ∫ B-dot dt [G]
            # High-pass: subtract running mean to remove integration drift
            from scipy.ndimage import uniform_filter1d
            hp_pts = max(1, int(round(200.0 / (dt_s * 1e3))))  # 200 ms window
            d = d - uniform_filter1d(d, size=hp_pts, mode='nearest')
            name = name + 'D→∫'

        if t_ref is None:
            t_ref = t.copy()
        if t_min_ms is not None or t_max_ms is not None:
            lo = t_min_ms if t_min_ms is not None else t[0]
            hi = t_max_ms if t_max_ms is not None else t[-1]
            mask = (t >= lo) & (t <= hi)
            t = t[mask]
            d = d[mask]
        signals.append(d)
        good_names.append(name)
        good_angles.append(phi)

    conn.closeAllTrees()

    if not signals:
        raise RuntimeError(f'No Mirnov signals fetched for shot {shot}')

    # Build common time axis from first probe's reference
    if t_ref is None:
        t_ref = np.linspace(0, 1, len(signals[0]))
    if t_min_ms is not None or t_max_ms is not None:
        lo = t_min_ms if t_min_ms is not None else t_ref[0]
        hi = t_max_ms if t_max_ms is not None else t_ref[-1]
        t_ref = t_ref[(t_ref >= lo) & (t_ref <= hi)]

    n_t_min = min(len(s) for s in signals)
    data = np.array([s[:n_t_min] for s in signals])   # (n_probes, n_t)
    t_ms = t_ref[:n_t_min]

    fs_khz = 1.0 / (float(np.mean(np.diff(t_ms))) * 1e-3) / 1e3
    sig_type = 'integrated B' if integrated else 'B-dot'
    print(f'  Fetched {len(good_names)}/{len(probe_names)} {array} probes '
          f'({sig_type}), t=[{t_ms[0]:.0f},{t_ms[-1]:.0f}] ms, fs={fs_khz:.0f} kHz')

    return data, t_ms, np.array(good_angles), good_names


def fetch_surfmn(shot, atlas='atlas.gat.com', t_min_ms=None, t_max_ms=None,
                 modes=None):
    """
    Fetch pre-computed SURFMN (m,n) mode amplitudes and radial locations
    from \\MHD::TOP.SURFMN.OUTPUT.ISLTABLE.

    B_M_N[n_idx, m_idx, t] where n_idx = n-1 (0-based) and m_idx = m (direct).
    ~20 ms cadence, full-shot coverage.

    Parameters
    ----------
    shot      : int
    atlas     : MDSplus server
    t_min_ms, t_max_ms : optional time crop [ms]
    modes     : list of (m, n) tuples to return; default [(2,1),(3,1),(4,1),(3,2)]

    Returns
    -------
    dict with keys:
      't_ms'   : (n_t,) time axis [ms]
      'amp'    : {(m,n): (n_t,) amplitude array}
      'rho'    : {(m,n): (n_t,) radial location (rho_N)}
    """
    if modes is None:
        modes = [(2, 1), (3, 1), (4, 1), (3, 2)]

    try:
        import MDSplus as mds
    except ImportError:
        raise RuntimeError(
            'MDSplus is not installed. It ships on the GA cluster and on '
            'conda-forge (conda install -c conda-forge mdsplus); fetching '
            'DIII-D data also requires network access to atlas.gat.com.'
        )

    conn = mds.Connection(atlas)
    conn.openTree('MHD', shot)
    base = '\\MHD::TOP.SURFMN.OUTPUT.ISLTABLE'

    bmn = np.array(conn.get(f'{base}:B_M_N').data())    # (n_max, m_max, n_t)
    loc = np.array(conn.get(f'{base}:LOC_M_N').data())
    t   = np.array(conn.get(f'DIM_OF({base}:B_M_N)').data())  # ms
    conn.closeAllTrees()

    mask = np.ones(len(t), dtype=bool)
    if t_min_ms is not None:
        mask &= t >= t_min_ms
    if t_max_ms is not None:
        mask &= t <= t_max_ms
    t = t[mask]

    amp_out = {}
    rho_out = {}
    for (m, n) in modes:
        n_idx = n - 1
        m_idx = m
        if n_idx < bmn.shape[0] and m_idx < bmn.shape[1]:
            amp_out[(m, n)] = bmn[n_idx, m_idx, mask]
            rho_out[(m, n)] = loc[n_idx, m_idx, mask]
        else:
            amp_out[(m, n)] = np.zeros(mask.sum())
            rho_out[(m, n)] = np.zeros(mask.sum())

    print(f'  SURFMN shot {shot}: {len(t)} time pts '
          f't=[{t[0]:.0f},{t[-1]:.0f}] ms, modes={modes}')
    return {'t_ms': t, 'amp': amp_out, 'rho': rho_out}


def plot_surfmn(results, shot_labels=None, onset_ms=None, n1rms_dict=None,
                modes=None, figsize=(11, 6)):
    """
    Plot SURFMN mode amplitudes and radial locations for one or more shots.

    Parameters
    ----------
    results    : dict {shot: fetch_surfmn output} or single fetch_surfmn output
    shot_labels: dict {shot: label string}
    onset_ms   : dict {shot: onset time [ms]} or scalar
    n1rms_dict : dict {shot: n1rms fetch_overview dict} — adds N1RMS panel if provided
    modes      : list of (m,n) to plot; default all modes in first result
    """
    if not isinstance(results, dict) or 't_ms' in results:
        results = {0: results}
        if shot_labels is None:
            shot_labels = {0: ''}

    shots = list(results.keys())
    if modes is None:
        modes = list(next(iter(results.values()))['amp'].keys())
    if shot_labels is None:
        shot_labels = {s: str(s) for s in shots}

    colors_mode = {(2,1): 'C3', (3,1): 'C1', (4,1): 'C4',
                   (3,2): 'C2', (1,1): 'C5'}
    ls_shot = ['-', '--', ':']

    n_panels = 3 if n1rms_dict else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    ax_amp, ax_rho = axes[0], axes[1]
    ax_rms = axes[2] if n_panels == 3 else None

    for i_s, shot in enumerate(shots):
        res = results[shot]
        t   = res['t_ms']
        ls  = ls_shot[i_s % len(ls_shot)]
        lbl_shot = shot_labels.get(shot, str(shot))

        for (m, n) in modes:
            amp = res['amp'].get((m, n), np.zeros_like(t))
            rho = res['rho'].get((m, n), np.zeros_like(t))
            col = colors_mode.get((m, n), f'C{m}')
            label = f'{lbl_shot}  {m}/{n}' if len(shots) > 1 else f'm/n={m}/{n}'
            ax_amp.plot(t, amp, color=col, ls=ls, lw=1.4, label=label)
            mask = rho > 0
            if mask.any():
                ax_rho.plot(t[mask], rho[mask], color=col, ls=ls, lw=1.4)

    ax_amp.set_ylabel('SURFMN amplitude [a.u.]')
    ax_amp.legend(fontsize=7, ncol=2)
    ax_amp.grid(True, alpha=0.3)

    ax_rho.set_ylabel(r'$\rho_N$ location')
    ax_rho.set_ylim(0, 1)
    ax_rho.axhline(0.59, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax_rho.grid(True, alpha=0.3)

    if ax_rms is not None:
        for i_s, shot in enumerate(shots):
            nd = n1rms_dict.get(shot, {})
            t_r = np.array(nd.get('time', []))
            v_r = np.array(nd.get('n1rms5', nd.get('n1rms', [])))
            if t_r.size and v_r.size:
                ax_rms.plot(t_r, v_r, lw=1.2, ls=ls_shot[i_s % 3],
                            label=shot_labels.get(shot, str(shot)))
        ax_rms.set_ylabel('N1RMS [G]')
        if ax_rms.get_legend_handles_labels()[0]:
            ax_rms.legend(fontsize=7)
        ax_rms.grid(True, alpha=0.3)
        ax_rms.set_xlabel('Time [ms]')
    else:
        ax_rho.set_xlabel('Time [ms]')

    # Onset markers
    if onset_ms is not None:
        ons = onset_ms if isinstance(onset_ms, dict) else {shots[0]: onset_ms}
        for i_s, shot in enumerate(shots):
            t_on = ons.get(shot)
            if t_on is not None:
                for ax in axes:
                    ax.axvline(t_on, color='k', ls='--', lw=0.9, alpha=0.6)

    shot_str = ' / '.join(str(s) for s in shots)
    ax_amp.set_title(f'Shot {shot_str} — SURFMN mode amplitudes', fontsize=10)
    fig.tight_layout()
    return fig


def fetch_ece(shot, atlas='atlas.gat.com', n_channels=40,
              t_min_ms=None, t_max_ms=None, bt0_T=None, r_axis_m=None,
              fast=True):
    """
    Fetch ECE Te channels from the 'ece' MDSplus tree.

    fast=True  (default): \\TECEF01…\\TECEF{n}  500 kHz — for fluctuation/burst analysis
    fast=False           : \\TECE01…\\TECE{n}   ~5 kHz  — for time-averaged Te profiles

    Radial positions computed from B_T(R) = B_T0 × R_axis / R at second harmonic.

    Parameters
    ----------
    shot       : int
    atlas      : MDSplus server
    n_channels : number of fixed-frequency channels to fetch (default 40)
    t_min_ms, t_max_ms : optional time crop [ms]
    bt0_T      : B_T at magnetic axis [T] for R_ece calculation
    r_axis_m   : R of magnetic axis [m]
    fast       : bool — True → TECEF (500 kHz); False → TECE (~5 kHz)

    Returns
    -------
    signals : ndarray (n_ch, n_t)  Te [keV]
    t_ms    : ndarray (n_t,)       time axis [ms]
    R_ece   : ndarray (n_ch,)      major radius per channel [m]
    freq_ghz: ndarray (n_ch,)      ECE frequency per channel [GHz]
    ch_idx  : list of int          channel indices that were successfully fetched
    """
    try:
        import MDSplus as mds
    except ImportError:
        raise RuntimeError(
            'MDSplus is not installed. It ships on the GA cluster and on '
            'conda-forge (conda install -c conda-forge mdsplus); fetching '
            'DIII-D data also requires network access to atlas.gat.com.'
        )

    freq_ghz, R_ece = ece_channel_radius(shot, bt0_T=bt0_T, r_axis_m=r_axis_m)

    conn = mds.Connection(atlas)
    conn.openTree('ece', shot)

    signals  = []
    good_ch  = []
    good_R   = []
    good_f   = []
    t_ref    = None

    prefix = 'TECEF' if fast else 'TECE'
    for i in range(1, n_channels + 1):
        name = f'\\{prefix}{i:02d}'
        try:
            d = np.array(conn.get(name).data(), dtype=float)
            t = np.array(conn.get(f'dim_of({name})').data(), dtype=float)
            if t.size < 2:
                continue
            if t_ref is None:
                t_ref = t.copy()
            if t_min_ms is not None or t_max_ms is not None:
                lo = t_min_ms if t_min_ms is not None else t[0]
                hi = t_max_ms if t_max_ms is not None else t[-1]
                mask = (t >= lo) & (t <= hi)
                t = t[mask]
                d = d[mask]
            signals.append(d)
            good_ch.append(i)
            good_R.append(R_ece[i - 1])
            good_f.append(freq_ghz[i - 1])
        except Exception as e:
            print(f'  Warning: ECE ch{i:02d} failed: {e}')

    conn.closeAllTrees()

    if not signals:
        raise RuntimeError(f'No ECE channels fetched for shot {shot}')

    if t_ref is None:
        t_ref = np.linspace(0, 1, len(signals[0]))
    if t_min_ms is not None or t_max_ms is not None:
        lo = t_min_ms if t_min_ms is not None else t_ref[0]
        hi = t_max_ms if t_max_ms is not None else t_ref[-1]
        t_ref = t_ref[(t_ref >= lo) & (t_ref <= hi)]

    n_t = min(len(s) for s in signals)
    data = np.array([s[:n_t] for s in signals])
    t_ms = t_ref[:n_t]
    fs_khz = 1.0 / (float(np.mean(np.diff(t_ms))) * 1e-3) / 1e3

    print(f'  Fetched {len(good_ch)}/{n_channels} ECE channels, '
          f't=[{t_ms[0]:.0f},{t_ms[-1]:.0f}] ms, fs={fs_khz:.0f} kHz, '
          f'R=[{min(good_R):.3f},{max(good_R):.3f}] m')

    return (data, t_ms,
            np.array(good_R),
            np.array(good_f),
            good_ch)


def ece_mode_location(ece_signals, t_ms, R_ece,
                       f_mode_khz, t_start_ms, t_end_ms,
                       df_band_khz=3.0, overview=None,
                       ref_signal=None, ref_t_ms=None):
    """
    Locate a rotating MHD mode radially from ECE Te oscillation amplitude.

    Two modes (controlled by ref_signal):

    ref_signal=None  [default — bandpass RMS]
      Bandpass each ECE channel at f_mode_khz ± df_band_khz/2, compute
      δTe/Te RMS over [t_start_ms, t_end_ms].

    ref_signal provided  [coherence mode]
      Compute the mean-squared coherence γ²(f_mode) between each ECE channel
      and the reference (e.g. N1RMS interpolated to the ECE timebase).
      Coherence is dimensionless (0–1) and insensitive to absolute Te level,
      removing the optical-depth bias that makes core channels dominate in
      the RMS method.  ref_t_ms must also be provided (timebase of ref_signal).

    Parameters
    ----------
    ece_signals : (n_ch, n_t) Te [keV]
    t_ms        : (n_t,) time [ms]
    R_ece       : (n_ch,) major radius per channel [m]
    f_mode_khz  : float, mode rotation frequency [kHz]
    t_start_ms, t_end_ms : time window for amplitude analysis [ms]
    df_band_khz : full bandpass width [kHz]
    overview    : dict from overview_ntm.pkl for rho estimate (optional)
    ref_signal  : (n_ref,) reference signal (e.g. N1RMS) — enables coherence mode
    ref_t_ms    : (n_ref,) timebase of ref_signal [ms]

    Returns
    -------
    dict with:
      R_ece        : (n_ch,) major radius array [m]
      rms_vs_R     : (n_ch,) metric vs R (δTe/Te in RMS mode; γ² in coherence mode)
      R_peak       : float, R at maximum metric [m]
      ch_peak      : int,   channel index (0-based) at peak
      rho_peak     : float or None, normalised rho at peak (outboard)
      f_mode_khz   : float
      t_start_ms, t_end_ms : float
      method       : 'rms' or 'coherence'
    """
    from scipy.signal import butter, filtfilt, coherence as sp_coherence

    dt_ms  = float(np.mean(np.diff(t_ms)))
    fs_hz  = 1e3 / dt_ms
    f_lo   = max(1.0, f_mode_khz - df_band_khz / 2.0)
    f_hi   = f_mode_khz + df_band_khz / 2.0
    nyq    = 0.5 * fs_hz
    lo_n, hi_n = f_lo * 1e3 / nyq, f_hi * 1e3 / nyq
    hi_n = min(hi_n, 0.999)

    b, a = butter(4, [lo_n, hi_n], btype='band')
    mask = (t_ms >= t_start_ms) & (t_ms <= t_end_ms)

    use_coherence = ref_signal is not None and ref_t_ms is not None
    rms_vs_R = np.zeros(len(R_ece))

    if use_coherence:
        # Interpolate reference onto ECE timebase; restrict to analysis window
        ref_interp = np.interp(t_ms, ref_t_ms, ref_signal)
        ref_win = ref_interp[mask] - np.mean(ref_interp[mask])
        nperseg = min(256, mask.sum() // 4)
        nperseg = max(nperseg, 16)
        for j in range(len(R_ece)):
            sig = ece_signals[j, mask]
            Te0 = float(np.median(sig))
            if Te0 < 0.05:
                continue
            sig_hp = sig - np.mean(sig)
            try:
                f_coh, coh = sp_coherence(sig_hp, ref_win, fs=fs_hz,
                                          nperseg=nperseg)
                # mean γ² over the mode frequency band
                band = (f_coh >= f_lo * 1e3) & (f_coh <= f_hi * 1e3)
                rms_vs_R[j] = float(np.mean(coh[band])) if band.any() else 0.0
            except Exception:
                rms_vs_R[j] = 0.0
        method = 'coherence'
    else:
        for j in range(len(R_ece)):
            sig = ece_signals[j]
            Te0 = float(np.median(sig))
            sig_hp = sig - Te0
            try:
                filt = filtfilt(b, a, sig_hp)
            except Exception:
                filt = sig_hp
            abs_rms = float(np.sqrt(np.mean(filt[mask] ** 2)))
            rms_vs_R[j] = abs_rms / Te0 if Te0 > 0.05 else 0.0
        method = 'rms'

    ch_peak = int(np.argmax(rms_vs_R))
    R_peak  = float(R_ece[ch_peak])

    # rho_N per channel: |R - R_axis| / a_minor (unsigned; override with EFIT in caller)
    rho_peak  = None
    rho_vs_R  = None
    r_ax_ref  = None
    a_min_ref = None
    if overview is not None:
        t_mid    = 0.5 * (t_start_ms + t_end_ms)
        r_ax_ref = float(np.interp(t_mid,
                                   np.array(overview.get('rmaxis', {}).get('time', [t_mid])),
                                   np.array(overview.get('rmaxis', {}).get('data', [1.69]))))
        a_min_ref = float(np.interp(t_mid,
                                    np.array(overview.get('aminor', {}).get('time', [t_mid])),
                                    np.array(overview.get('aminor', {}).get('data', [0.60]))))
        if a_min_ref > 0:
            rho_vs_R = np.abs(R_ece - r_ax_ref) / a_min_ref
            rho_peak = float(rho_vs_R[ch_peak])

    return {
        'R_ece':       R_ece,
        'rms_vs_R':    rms_vs_R,
        'R_peak':      R_peak,
        'ch_peak':     ch_peak,
        'rho_peak':    rho_peak,
        'rho_vs_R':    rho_vs_R,
        'r_axis_m':    r_ax_ref,
        'a_minor_m':   a_min_ref,
        'f_mode_khz':  f_mode_khz,
        't_start_ms':  t_start_ms,
        't_end_ms':    t_end_ms,
        'method':      method,
    }


def plot_ece_location(loc_result, shot=None, rho_q2=None, figsize=(9, 4)):
    """
    Two-panel ECE mode location plot.

    Left:  RMS amplitude vs major radius R  (marks peak channel)
    Right: RMS amplitude vs rho_N (outboard midplane), with q=2 surface
    """
    R    = loc_result['R_ece']
    rms  = loc_result['rms_vs_R']
    R_pk = loc_result['R_peak']
    rho  = loc_result.get('rho_peak')
    f0   = loc_result['f_mode_khz']
    t0   = loc_result['t_start_ms']
    t1   = loc_result['t_end_ms']

    sort_r = np.argsort(R)
    R_s   = R[sort_r]
    rms_s = rms[sort_r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    title = (f'Shot {shot} — ' if shot else '') + \
            f'ECE mode location  f={f0:.1f} kHz  t=[{t0:.0f},{t1:.0f}] ms'
    fig.suptitle(title, fontsize=9)

    method = loc_result.get('method', 'rms')
    y_label = 'γ² at f_mode (coherence)' if method == 'coherence' else 'δTe/Te (%)'
    y_scale = 1.0 if method == 'coherence' else 1e2

    # Left: metric vs R
    ax1.plot(R_s, rms_s * y_scale, 'o-', lw=1.2, ms=4)
    ax1.axvline(R_pk, color='r', ls='--', lw=1, label=f'R={R_pk:.3f} m')
    ax1.set_xlabel('Major radius R (m)')
    ax1.set_ylabel(y_label)
    ax1.legend(fontsize=8)
    ax1.grid(False)

    # Right: RMS vs rho_N (outboard midplane)
    rho_arr = loc_result.get('rho_vs_R')
    if rho_arr is not None:
        sort_rho  = np.argsort(rho_arr)
        rho_s     = rho_arr[sort_rho]
        rms_rho_s = rms[sort_rho]
        rho_pk    = rho
        ax2.plot(rho_s, rms_rho_s * y_scale, 'o-', lw=1.2, ms=4, color='C0')
        if rho_pk is not None:
            ax2.axvline(rho_pk, color='r', ls='--', lw=1, label=f'ρ_N={rho_pk:.2f}')
        if rho_q2 is not None:
            ax2.axvline(rho_q2, color='k', ls=':', lw=1.2, label=f'q=2  ρ={rho_q2:.2f}')
        ax2.set_xlabel('ρ_N  (outboard midplane)')
        ax2.set_ylabel(y_label)
        ax2.legend(fontsize=8)
        ax2.grid(False)
    else:
        ax2.plot(np.arange(len(rms)), rms * y_scale, 'o-', lw=1.2, ms=4)
        ax2.set_xlabel('ECE channel index')
        ax2.set_ylabel('δTe/Te (%)')
        ax2.grid(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── EFIT q-map ────────────────────────────────────────────────────────────────

def fetch_efit_qmap(shot, t_ms, atlas='atlas.gat.com', tree='EFIT02ER'):
    """
    Fetch 2-D q map from EFIT02ER at the time slice closest to t_ms.

    Reads PSIRZ (nt, nZ, nR), QPSI (nt, nq), SSIMAG/SSIBRY (nt,),
    RMAXIS/ZMAXIS/RBBBS/ZBBBS from the EFIT02ER MDSplus tree.

    Returns
    -------
    dict with keys:
      R_grid   : (nR,)     major radius grid [m]
      Z_grid   : (nZ,)     height grid [m]
      psi_n    : (nZ, nR)  normalised poloidal flux at t_actual
      q_2d     : (nZ, nR)  q value mapped onto spatial grid
      q_psi    : (nq,)     q profile on uniform psi_N = linspace(0,1,nq)
      psi_1d   : (nq,)     psi_N grid for q_psi
      t_actual : float     actual time of slice [ms]
      R_axis   : float     R of magnetic axis [m]
      Z_axis   : float     Z of magnetic axis [m]
      R_bdy    : (nb,)     LCFS R [m]
      Z_bdy    : (nb,)     LCFS Z [m]
    """
    try:
        import MDSplus as mds
    except ImportError:
        raise RuntimeError(
            'MDSplus is not installed. It ships on the GA cluster and on '
            'conda-forge (conda install -c conda-forge mdsplus); fetching '
            'DIII-D data also requires network access to atlas.gat.com.'
        )
    from scipy.interpolate import interp1d

    base = f'\\{tree}::TOP.RESULTS.GEQDSK:'
    conn = mds.Connection(atlas)
    conn.openTree(tree, shot)

    psirz  = np.array(conn.get(f'{base}PSIRZ').data(),  dtype=float)   # (nt, nZ, nR)
    r_arr  = np.array(conn.get(f'dim_of({base}PSIRZ, 0)').data(), dtype=float)
    z_arr  = np.array(conn.get(f'dim_of({base}PSIRZ, 1)').data(), dtype=float)
    t_arr  = np.array(conn.get(f'dim_of({base}PSIRZ, 2)').data(), dtype=float)
    qpsi   = np.array(conn.get(f'{base}QPSI').data(),   dtype=float)   # (nt, nq)
    ssimag = np.array(conn.get(f'{base}SSIMAG').data(), dtype=float)   # (nt,)
    ssibry = np.array(conn.get(f'{base}SSIBRY').data(), dtype=float)   # (nt,)
    rmaxis = np.array(conn.get(f'{base}RMAXIS').data(), dtype=float)
    zmaxis = np.array(conn.get(f'{base}ZMAXIS').data(), dtype=float)
    rbbbs  = np.array(conn.get(f'{base}RBBBS').data(),  dtype=float)   # (nt, nb)
    zbbbs  = np.array(conn.get(f'{base}ZBBBS').data(),  dtype=float)
    conn.closeAllTrees()

    ti = int(np.argmin(np.abs(t_arr - t_ms)))
    t_actual = float(t_arr[ti])

    psi_slice = psirz[ti]                             # (nZ, nR)
    dpsi      = float(ssibry[ti] - ssimag[ti])
    psi_n     = (psi_slice - ssimag[ti]) / dpsi       # (nZ, nR)  0=axis, 1=LCFS

    nq      = qpsi.shape[1]
    psi_1d  = np.linspace(0.0, 1.0, nq)
    q_row   = qpsi[ti]                               # (nq,) from axis to boundary
    q_func  = interp1d(psi_1d, q_row, kind='linear',
                       bounds_error=False, fill_value=(q_row[0], q_row[-1]))
    q_2d    = q_func(np.clip(psi_n, 0.0, 1.0))       # (nZ, nR)

    print(f'  EFIT {tree} t={t_actual:.0f} ms  '
          f'q_axis={q_row[0]:.2f}  q(psi_N=0.9)={q_func(0.9):.2f}  '
          f'R_axis={rmaxis[ti]:.3f} m')

    return {
        'R_grid':   r_arr,
        'Z_grid':   z_arr,
        'psi_n':    psi_n,
        'q_2d':     q_2d,
        'q_psi':    q_row,
        'psi_1d':   psi_1d,
        't_actual': t_actual,
        'R_axis':   float(rmaxis[ti]),
        'Z_axis':   float(zmaxis[ti]),
        'R_bdy':    rbbbs[ti],
        'Z_bdy':    zbbbs[ti],
    }


def plot_efit_qmap(qmap, ece_R_peak=None, ece_rho_peak=None,
                   q_contours=(1.5, 2.0, 2.5, 3.0),
                   shot=None, figsize=(6, 7)):
    """
    Plot 2-D q map from EFIT with ECE mode location overlaid.

    Filled contours of q on (R, Z), LCFS boundary, magnetic axis.
    ECE midplane sightline (Z=0) and ECE peak R marked.

    Parameters
    ----------
    qmap         : dict returned by fetch_efit_qmap
    ece_R_peak   : float  — major radius of ECE amplitude peak [m]
    ece_rho_peak : float  — normalised rho for annotation (optional)
    q_contours   : iterable of q values to draw as labelled contour lines
    """
    R   = qmap['R_grid']
    Z   = qmap['Z_grid']
    q2d = qmap['q_2d']
    t   = qmap['t_actual']

    fig, ax = plt.subplots(figsize=figsize)
    title = (f'Shot {shot} — ' if shot else '') + f'q map  t={t:.0f} ms  (EFIT02ER)'
    ax.set_title(title, fontsize=9)

    # Filled q contour (clipped for colour clarity)
    q_plot = np.clip(q2d, 0.5, 5.0)
    cf = ax.contourf(R, Z, q_plot, levels=40, cmap='RdYlBu_r', alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('q', fontsize=9)

    # Named q contours
    cs = ax.contour(R, Z, q2d, levels=list(q_contours),
                    colors='k', linewidths=[2.0 if q == 2.0 else 0.8
                                            for q in q_contours])
    ax.clabel(cs, fmt='q=%.1f', fontsize=7, inline=True)

    # LCFS boundary
    ax.plot(qmap['R_bdy'], qmap['Z_bdy'], 'k-', lw=1.5, label='LCFS')

    # Magnetic axis
    ax.plot(qmap['R_axis'], qmap['Z_axis'], '+', color='k', ms=10, mew=2)

    # ECE midplane sightline
    ax.axhline(0.0, color='royalblue', ls='--', lw=1.2, label='ECE sightline  (Z=0)')

    # ECE peak radius
    if ece_R_peak is not None:
        q_at_peak = float(np.interp(ece_R_peak, R,
                                    q2d[int(np.argmin(np.abs(Z - 0.0))), :]))
        lbl = f'ECE peak  R={ece_R_peak:.3f} m  q={q_at_peak:.2f}'
        if ece_rho_peak is not None:
            lbl += f'  ρ={ece_rho_peak:.2f}'
        ax.axvline(ece_R_peak, color='r', ls='-', lw=1.5, label=lbl)
        ax.plot(ece_R_peak, 0.0, 'r*', ms=12, zorder=6)
        print(f'  q at ECE peak (R={ece_R_peak:.3f} m, Z=0): {q_at_peak:.3f}')

    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _freq_smooth(x, nsmooth):
    """Uniform running average over nsmooth bins along last axis."""
    if nsmooth < 2:
        return x
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x.real, size=nsmooth, axis=-1) + \
           1j * uniform_filter1d(x.imag, size=nsmooth, axis=-1) \
           if np.iscomplexobj(x) else \
           uniform_filter1d(x, size=nsmooth, axis=-1)


def _c95(nsmooth):
    """95% coherence confidence level (Fisher z-transform, IDL formula)."""
    if nsmooth < 2:
        return 0.0
    z = 1.96 / np.sqrt(2.0 * nsmooth - 2.0)
    tanh_z = np.tanh(z)
    return float(tanh_z ** 2)


# ── Core mode analysis ─────────────────────────────────────────────────────────

def mode_spectrogram(signals, t_ms, phi_deg,
                     dt_window_ms=4.0, overlap_frac=0.75,
                     f_min_khz=5.0, f_max_khz=200.0,
                     f_smooth_khz=2.0,
                     n_range=(-5, 5),
                     remove_dc=True):
    """
    Compute power spectrogram and toroidal mode number spectrogram.

    Algorithm:
    1. Divide each probe signal into overlapping Hann-windowed time slices.
    2. FFT each slice; optionally smooth spectra in frequency (f_smooth_khz).
    3. Matched-filter mode number fit at each (time, freq) bin:
         A_n = |Σ_j F_j · exp(-i·n·φ_j)| / N_probes
    4. Dominant n = argmax(A_n); coherence = A_dom / ΣA_all.
    5. Compute 95% coherence confidence level (c95) from nsmooth.

    Parameters
    ----------
    signals      : (n_probes, n_t)
    t_ms         : (n_t,) time in ms
    phi_deg      : (n_probes,) toroidal angles in degrees
    dt_window_ms : FFT window length [ms]
    overlap_frac : fractional window overlap (0–1)
    f_min_khz, f_max_khz : analysis frequency band [kHz]
    f_smooth_khz : frequency smoothing bandwidth [kHz]; 0 disables smoothing
    n_range      : (n_min, n_max) toroidal mode numbers to test
    remove_dc    : subtract window mean before FFT

    Returns
    -------
    dict with keys:
      t_win_ms   : (n_win,)        window centre times [ms]
      freq_khz   : (n_freq,)       frequency axis [kHz]
      power      : (n_win, n_freq) total power [G²/kHz]
      n_dominant : (n_win, n_freq) dominant toroidal mode number
      coherence  : (n_win, n_freq) matched-filter coherence (0–1)
      mode_amp   : dict {n: (n_win, n_freq)} mode amplitude [G/√kHz]
      rms_vs_time: dict {n: (n_win,)} RMS per mode [G]
      c95        : float  95% coherence confidence level
      nsmooth    : int    frequency bins smoothed
      f_min_khz, f_max_khz, n_range, phi_deg, nwin, fs_khz : metadata
    """
    n_probes, n_t = signals.shape
    dt_ms  = float(np.mean(np.diff(t_ms)))
    fs_khz = 1.0 / dt_ms

    nwin   = int(round(dt_window_ms / dt_ms))
    nstep  = max(1, int(round(nwin * (1 - overlap_frac))))
    phi_rad = np.deg2rad(phi_deg)

    freqs_khz = np.fft.rfftfreq(nwin, d=dt_ms)
    df_khz    = freqs_khz[1] if len(freqs_khz) > 1 else 1.0
    f_mask    = (freqs_khz >= f_min_khz) & (freqs_khz <= f_max_khz)
    freqs_out = freqs_khz[f_mask]

    nsmooth = max(1, int(round(f_smooth_khz / df_khz))) if f_smooth_khz > 0 else 1
    c95_val = _c95(nsmooth)

    win     = np.hanning(nwin)
    n_modes = np.arange(n_range[0], n_range[1] + 1)

    # Matched-filter steering vectors: (n_modes, n_probes)
    phase_vectors = np.exp(-1j * np.outer(n_modes, phi_rad))

    starts   = np.arange(0, n_t - nwin + 1, nstep)
    n_win    = len(starts)
    t_win_ms = t_ms[starts + nwin // 2]

    n_freq     = freqs_out.size
    power      = np.zeros((n_win, n_freq))
    n_dominant = np.zeros((n_win, n_freq), dtype=int)
    coherence  = np.zeros((n_win, n_freq))
    mode_amp   = {int(n): np.zeros((n_win, n_freq)) for n in n_modes}

    for iw, i0 in enumerate(starts):
        seg = signals[:, i0:i0 + nwin].copy()
        if remove_dc:
            seg -= seg.mean(axis=1, keepdims=True)

        # FFT: (n_probes, n_rfft)
        F_full = np.fft.rfft(seg * win[None, :], axis=1)

        # Smooth auto-power in frequency, then extract band
        auto = np.abs(F_full) ** 2                        # (n_probes, n_rfft)
        if nsmooth > 1:
            from scipy.ndimage import uniform_filter1d
            auto = uniform_filter1d(auto, size=nsmooth, axis=1)

        # Power: mean auto-power over probes, normalised to G²/kHz
        power[iw] = auto.mean(axis=0)[f_mask] / fs_khz

        # Matched-filter amplitude: smooth F in frequency before filtering
        F = F_full[:, f_mask]                             # (n_probes, n_freq)
        if nsmooth > 1:
            from scipy.ndimage import uniform_filter1d
            F = (uniform_filter1d(F.real, size=nsmooth, axis=1) +
                 1j * uniform_filter1d(F.imag, size=nsmooth, axis=1))

        # A[n_modes, n_freq] = |phase_vectors @ F| / n_probes
        A = np.abs(phase_vectors @ F) / n_probes

        for j, n in enumerate(n_modes):
            mode_amp[int(n)][iw] = A[j]

        best            = np.argmax(A, axis=0)
        n_dominant[iw]  = n_modes[best]
        total_amp        = A.sum(axis=0)
        coherence[iw]   = A[best, np.arange(n_freq)] / (total_amp + 1e-30)

    # RMS per mode: integrate mode_amp² over frequency
    rms_vs_time = {int(n): np.sqrt(np.trapz(mode_amp[int(n)] ** 2, freqs_out, axis=1))
                   for n in n_modes}

    return {
        't_win_ms':   t_win_ms,
        't_sig_ms':   (float(t_ms[0]), float(t_ms[-1])),
        'freq_khz':   freqs_out,
        'power':      power,
        'n_dominant': n_dominant,
        'coherence':  coherence,
        'mode_amp':   mode_amp,
        'rms_vs_time': rms_vs_time,
        'c95':        c95_val,
        'nsmooth':    nsmooth,
        'f_min_khz':  f_min_khz,
        'f_max_khz':  f_max_khz,
        'n_range':    n_range,
        'phi_deg':    phi_deg,
        'nwin':       nwin,
        'fs_khz':     fs_khz,
    }


def mode_fit_timeslice(signals, t_ms, phi_deg, t0_ms, f0_khz,
                       dt_window_ms=4.0, f_smooth_khz=2.0, n_range=(1, 5)):
    """
    Coherence-weighted phase fit at a single (t0, f0) point.

    Matches IDL phase_fit.pro: weights phase residuals by 1/sigma where
    sigma = sqrt((1/coherence - 1) / nsmooth), then returns the weighted
    chi-squared for each candidate mode number n.

    Parameters
    ----------
    signals, t_ms, phi_deg : from fetch_mirnov
    t0_ms, f0_khz          : target time and frequency
    dt_window_ms           : FFT window length [ms]
    f_smooth_khz           : frequency smoothing bandwidth [kHz]
    n_range                : (n_min, n_max) mode numbers to test

    Returns
    -------
    dict with keys: phi_deg, power_vs_phi, phase_vs_phi, coherence_vs_phi,
                    n_modes, chi2_vs_n (weighted), n_best, t0_ms, f0_khz,
                    fit_curve_phi, fit_curve_phase (for the best n)
    """
    dt_ms   = float(np.mean(np.diff(t_ms)))
    nwin    = int(round(dt_window_ms / dt_ms))
    fs_khz  = 1.0 / dt_ms
    df_khz  = 1.0 / (nwin * dt_ms)
    nsmooth = max(1, int(round(f_smooth_khz / df_khz))) if f_smooth_khz > 0 else 1

    i0 = np.searchsorted(t_ms, t0_ms) - nwin // 2
    i0 = max(0, min(i0, len(t_ms) - nwin))

    seg = signals[:, i0:i0 + nwin].copy()
    seg -= seg.mean(axis=1, keepdims=True)

    F_full    = np.fft.rfft(seg * np.hanning(nwin)[None, :], axis=1)
    freqs_khz = np.fft.rfftfreq(nwin, d=dt_ms)
    i_f       = np.argmin(np.abs(freqs_khz - f0_khz))

    # Smooth in frequency around target bin
    if nsmooth > 1:
        from scipy.ndimage import uniform_filter1d
        F_sm = (uniform_filter1d(F_full.real, size=nsmooth, axis=1) +
                1j * uniform_filter1d(F_full.imag, size=nsmooth, axis=1))
        auto_sm = uniform_filter1d(np.abs(F_full) ** 2, size=nsmooth, axis=1)
    else:
        F_sm    = F_full
        auto_sm = np.abs(F_full) ** 2

    C = F_sm[:, i_f]          # complex amplitude at f0 for each probe

    # Cross-coherence with probe 0 (reference), matching IDL array_spec
    cross_auto  = np.abs(F_sm[:, i_f]) ** 2          # auto per probe at f0
    cross_cross = np.abs(F_sm[:, i_f] * np.conj(F_sm[0, i_f])) ** 2
    coh_vs_phi  = cross_cross / (cross_auto * cross_auto[0] + 1e-30)
    coh_vs_phi  = np.clip(coh_vs_phi, 1e-6, 1.0)

    # Phase uncertainty weights (IDL phase_fit.pro formula)
    sigma = np.sqrt((1.0 / coh_vs_phi - 1.0) / max(nsmooth, 1))
    sigma = np.maximum(sigma, 1e-3)   # floor to avoid zero

    phi_rad = np.deg2rad(phi_deg)
    n_modes = np.arange(n_range[0], n_range[1] + 1)

    chi2 = np.zeros(len(n_modes))
    w = 1.0 / sigma
    for j, n in enumerate(n_modes):
        resid = np.angle(C * np.exp(-1j * n * phi_rad))   # rad, wrapped in [-pi, pi]
        # Remove the free global phase offset (circular weighted mean)
        phi_0 = np.angle(np.sum(np.exp(1j * resid) * w))
        resid  = np.angle(np.exp(1j * (resid - phi_0)))
        chi2[j] = float(np.sum(resid ** 2 * w) / (np.sum(w) + 1e-30))

    n_best = n_modes[np.argmin(chi2)]

    # Best-fit phase line: use the fitted intercept so the line aligns with data
    resid_best = np.angle(C * np.exp(-1j * n_best * phi_rad))
    phi_0_best = np.angle(np.sum(np.exp(1j * resid_best) * w))
    phi_fit    = np.linspace(phi_deg.min() - 20, phi_deg.max() + 20, 200)
    phase_fit  = np.degrees(np.angle(np.exp(1j * (n_best * np.deg2rad(phi_fit) + phi_0_best))))

    return {
        'phi_deg':          phi_deg,
        'power_vs_phi':     np.abs(C) ** 2,
        'phase_vs_phi':     np.angle(C),
        'coherence_vs_phi': coh_vs_phi,
        'n_modes':          n_modes,
        'chi2_vs_n':        chi2,
        'n_best':           int(n_best),
        't0_ms':            float(t_ms[i0 + nwin // 2]),
        'f0_khz':           float(freqs_khz[i_f]),
        'fit_curve_phi':    phi_fit,
        'fit_curve_phase':  phase_fit,
        'nsmooth':          nsmooth,
    }


def mode_fit_lsq(C, phi_deg, n_range=(-5, 5)):
    """
    Multi-harmonic least-squares fit to probe complex amplitudes.

    Matches IDL slice_fit.pro: fits all harmonics n=nmin..nmax simultaneously
    via matrix inversion rather than independently.

    Parameters
    ----------
    C       : (n_probes,) complex FFT amplitudes at one (t, f) point
    phi_deg : (n_probes,) toroidal angles [deg]
    n_range : (n_min, n_max) harmonics to fit simultaneously

    Returns
    -------
    dict with:
      coeffs    : dict {n: (cos_coeff, sin_coeff)} for each n
      amplitude : dict {n: amplitude} (sqrt(cos² + sin²))
      phase_deg : dict {n: phase [deg]}
      fit_phi   : (200,) angle axis for fitted curve [deg]
      fit_B     : (200,) fitted magnetic perturbation (normalised)
      n_dominant: int, n with largest amplitude
    """
    phi_rad = np.deg2rad(phi_deg)
    n_min, n_max = n_range
    n_list  = list(range(n_min, n_max + 1))
    n_basis = len(n_list) * 2   # cos + sin per harmonic

    # Design matrix A: (n_probes, n_basis) — real-valued fitting of Re(C)
    # Using |C| projected onto each harmonic via cos/sin basis
    n_probes = len(C)
    A = np.zeros((n_probes, n_basis))
    for k, n in enumerate(n_list):
        A[:, 2 * k]     = np.cos(n * phi_rad)
        A[:, 2 * k + 1] = np.sin(n * phi_rad)

    y = np.abs(C) * np.cos(np.angle(C))   # Re(C)

    # Least-squares: coeffs = (AᵀA)⁻¹ Aᵀ y
    ATA = A.T @ A
    try:
        cc = np.linalg.solve(ATA, A.T @ y)
    except np.linalg.LinAlgError:
        cc = np.linalg.lstsq(A, y, rcond=None)[0]

    coeffs    = {}
    amplitude = {}
    phase_deg = {}
    for k, n in enumerate(n_list):
        a_cos, a_sin = cc[2 * k], cc[2 * k + 1]
        coeffs[n]    = (float(a_cos), float(a_sin))
        amplitude[n] = float(np.sqrt(a_cos ** 2 + a_sin ** 2))
        phase_deg[n] = float(np.degrees(np.arctan2(a_sin, a_cos)))

    # Fitted curve
    phi_fit = np.linspace(-360, 360, 200)
    phi_fit_rad = np.deg2rad(phi_fit)
    B_fit = np.zeros(200)
    for k, n in enumerate(n_list):
        a_cos, a_sin = cc[2 * k], cc[2 * k + 1]
        B_fit += a_cos * np.cos(n * phi_fit_rad) + a_sin * np.sin(n * phi_fit_rad)

    n_dom = max(n_list, key=lambda n: amplitude[n])

    return {
        'coeffs':     coeffs,
        'amplitude':  amplitude,
        'phase_deg':  phase_deg,
        'fit_phi':    phi_fit,
        'fit_B':      B_fit,
        'n_dominant': int(n_dom),
    }


# ── SVD / MUSIC mode analysis ──────────────────────────────────────────────────

def mode_svd_spectrogram(signals, t_ms, phi_deg,
                          dt_window_ms=4.0, overlap_frac=0.75,
                          n_avg=8,
                          f_min_khz=5.0, f_max_khz=200.0,
                          f_smooth_khz=2.0,
                          n_range=(-5, 5),
                          n_src=1,
                          music=True,
                          remove_dc=True):
    """
    SVD-based mode number spectrogram via cross-spectral matrix (CSM) analysis.

    At each (time, frequency) bin, averages `n_avg` neighbouring FFT snapshots
    to form the Hermitian CSM:

        S[j,k] = (1/N) Σ_snapshots  F_j · conj(F_k)

    Eigendecompose S = U Λ Uᴴ (Hermitian, so all-real eigenvalues).
    The dominant eigenvector u₁ represents the coherent spatial mode with the
    most power.  Mode number is identified by matched-filter projection of u₁
    onto toroidal steering vectors.

    Optionally computes the MUSIC pseudospectrum using the noise subspace:

        P_MUSIC(n) = 1 / (1 − |aₙᴴ u₁|²)   [dB]

    which has a sharper peak at the true mode number than the matched filter.

    Parameters
    ----------
    signals      : (n_probes, n_t)
    t_ms         : (n_t,) time in ms
    phi_deg      : (n_probes,) toroidal angles in degrees
    dt_window_ms : FFT window length [ms]
    overlap_frac : fractional window overlap (0–1)
    n_avg        : number of FFT snapshots to average for each CSM estimate
    f_min_khz, f_max_khz : analysis frequency band [kHz]
    f_smooth_khz : frequency smoothing bandwidth [kHz] applied to each FFT
    n_range      : (n_min, n_max) toroidal mode numbers
    n_src        : number of assumed signal sources (noise subspace = n_probes - n_src)
    music        : if True, compute MUSIC pseudospectrum
    remove_dc    : subtract window mean before FFT

    Returns
    -------
    dict with keys (compatible with plot_modespec):
      t_win_ms     : (n_win,)              window centre times [ms]
      freq_khz     : (n_freq,)             frequency axis [kHz]
      power        : (n_win, n_freq)       total power = trace(CSM)/n_probes [G²/kHz]
      n_dominant   : (n_win, n_freq)       mode number from dominant eigenvector
      coherence    : (n_win, n_freq)       λ₁/Σλ — fractional power in dominant mode
      mode_amp     : dict {n: (n_win, n_freq)}  matched-filter amplitude on u₁
      rms_vs_time  : dict {n: (n_win,)}    RMS per mode [G]
      eigenvalues  : (n_win, n_freq, n_probes)  all eigenvalues, descending
      music_pseudo : (n_win, n_freq, n_modes)   MUSIC pseudospectrum [dB] or None
      c95, nsmooth, f_min_khz, f_max_khz, n_range, phi_deg, nwin, fs_khz, n_avg
    """
    n_probes, n_t = signals.shape
    dt_ms   = float(np.mean(np.diff(t_ms)))
    fs_khz  = 1.0 / dt_ms

    nwin    = int(round(dt_window_ms / dt_ms))
    nstep   = max(1, int(round(nwin * (1 - overlap_frac))))
    phi_rad = np.deg2rad(phi_deg)

    freqs_khz = np.fft.rfftfreq(nwin, d=dt_ms)
    df_khz    = freqs_khz[1] if len(freqs_khz) > 1 else 1.0
    f_mask    = (freqs_khz >= f_min_khz) & (freqs_khz <= f_max_khz)
    freqs_out = freqs_khz[f_mask]
    n_freq    = freqs_out.size

    nsmooth = max(1, int(round(f_smooth_khz / df_khz))) if f_smooth_khz > 0 else 1
    c95_val = _c95(nsmooth * n_avg)   # effective smoothing includes snapshot averaging

    win     = np.hanning(nwin)
    n_modes = np.arange(n_range[0], n_range[1] + 1)

    # Conjugate steering vectors for matched filter: exp(-i·n·φ)/√N
    # |aₙᴴ u₁| = |Σ_j exp(-i·n·φ_j)·u₁_j| — maximum when n equals the true mode number
    a_norm = np.exp(-1j * np.outer(n_modes, phi_rad)) / np.sqrt(n_probes)

    starts   = np.arange(0, n_t - nwin + 1, nstep)
    n_win    = len(starts)
    t_win_ms = t_ms[starts + nwin // 2]

    # ── Step 1: compute all FFT snapshots ─────────────────────────────────────
    # F_all: (n_win, n_probes, n_freq) complex
    F_all = np.zeros((n_win, n_probes, n_freq), dtype=complex)
    for iw, i0 in enumerate(starts):
        seg = signals[:, i0:i0 + nwin].copy()
        if remove_dc:
            seg -= seg.mean(axis=1, keepdims=True)
        F_full = np.fft.rfft(seg * win[None, :], axis=1)
        if nsmooth > 1:
            from scipy.ndimage import uniform_filter1d
            F_full = (uniform_filter1d(F_full.real, size=nsmooth, axis=1) +
                      1j * uniform_filter1d(F_full.imag, size=nsmooth, axis=1))
        F_all[iw] = F_full[:, f_mask]

    # ── Step 2: SVD / MUSIC at each output time via averaged CSM ──────────────
    half_avg = n_avg // 2
    power      = np.zeros((n_win, n_freq))
    n_dominant = np.zeros((n_win, n_freq), dtype=int)
    coherence  = np.zeros((n_win, n_freq))
    eigenvalues= np.zeros((n_win, n_freq, n_probes))
    mode_amp   = {int(n): np.zeros((n_win, n_freq)) for n in n_modes}
    music_pseudo = np.zeros((n_win, n_freq, len(n_modes))) if music else None

    for iw in range(n_win):
        # Snapshot block centred on current window
        i_lo = max(0, iw - half_avg)
        i_hi = min(n_win, iw + half_avg + 1)
        snap = F_all[i_lo:i_hi]              # (n_snap, n_probes, n_freq)
        n_snap = snap.shape[0]

        # CSM[j,l] = E[F_j · conj(F_l)]: (n_freq, n_probes, n_probes)
        S = np.moveaxis(snap, 2, 0)          # (n_freq, n_snap, n_probes)
        CSM = np.einsum('fkj,fkl->fjl', S, S.conj()) / n_snap

        # Eigendecompose (eigh: ascending order, real eigenvalues)
        vals, vecs = np.linalg.eigh(CSM)     # (n_freq, n_probes), (n_freq, n_probes, n_probes)
        vals = np.flip(vals, axis=-1)        # descending
        vecs = np.flip(vecs, axis=-1)        # dominant first

        eigenvalues[iw] = vals
        lambda_sum = vals.sum(axis=-1)       # (n_freq,)
        power[iw]  = np.real(lambda_sum) / (n_probes * fs_khz)
        coherence[iw] = np.real(vals[:, 0]) / (np.real(lambda_sum) + 1e-30)

        # Dominant eigenvector: u1 shape (n_freq, n_probes)
        u1 = vecs[:, :, 0]

        # Matched filter on u1: A[n_modes, n_freq]
        # a_norm: (n_modes, n_probes), u1.T: (n_probes, n_freq)
        A = np.abs(a_norm @ u1.T)            # (n_modes, n_freq)
        for j, n in enumerate(n_modes):
            mode_amp[int(n)][iw] = A[j]
        best = np.argmax(A, axis=0)
        n_dominant[iw] = n_modes[best]

        # MUSIC pseudospectrum
        if music:
            # Signal subspace: first n_src eigenvectors
            # |aₙᴴ U_sig|² = Σ_{k<n_src} |aₙ · u_k|²
            # MUSIC: 1 / (1 - |aₙᴴ U_sig|²)  [assumes normalised a]
            U_sig = vecs[:, :, :n_src]       # (n_freq, n_probes, n_src)
            # proj[n, f, k] = a_norm[n,:] @ U_sig[f,:,k]
            # |aₙᴴ U_sig|²: a_norm already has exp(-i·n·φ), so use directly
            proj = np.einsum('np,fpk->nfk', a_norm, U_sig)
            sig_power = np.sum(np.abs(proj) ** 2, axis=-1)  # (n_modes, n_freq)
            denom = np.clip(1.0 - sig_power, 1e-10, None)
            music_db = 10.0 * np.log10(1.0 / denom)
            music_pseudo[iw] = music_db.T    # (n_freq, n_modes)

    # ── RMS per mode vs time ──────────────────────────────────────────────────
    rms_vs_time = {int(n): np.sqrt(np.trapz(mode_amp[int(n)] ** 2, freqs_out, axis=1))
                   for n in n_modes}

    return {
        't_win_ms':    t_win_ms,
        't_sig_ms':    (float(t_ms[0]), float(t_ms[-1])),
        'freq_khz':    freqs_out,
        'power':       power,
        'n_dominant':  n_dominant,
        'coherence':   coherence,
        'mode_amp':    mode_amp,
        'rms_vs_time': rms_vs_time,
        'eigenvalues': eigenvalues,
        'music_pseudo': music_pseudo,
        'c95':         c95_val,
        'nsmooth':     nsmooth,
        'n_avg':       n_avg,
        'n_src':       n_src,
        'f_min_khz':   f_min_khz,
        'f_max_khz':   f_max_khz,
        'n_range':     n_range,
        'phi_deg':     phi_deg,
        'nwin':        nwin,
        'fs_khz':      fs_khz,
    }


def plot_svd(result, shot=None, n1rms_dict=None,
             coh_thresh=None, onset_ms=None, figsize=(13, 12)):
    """
    Five-panel SVD mode analysis plot.

    Panel 1: Power spectrogram (trace of CSM) [dB]
    Panel 2: Coherence λ₁/Σλ with c95 threshold line
    Panel 3: Dominant mode number (masked by coherence)
    Panel 4: MUSIC pseudospectrum at each time (summed over frequency band)
             — shows which mode number dominates vs time [dB]
    Panel 5: N1RMS (optional)
    """
    t    = result['t_win_ms']
    f    = result['freq_khz']
    pw   = result['power']
    coh  = result['coherence']
    nd   = result['n_dominant']
    rms  = result['rms_vs_time']
    c95  = result.get('c95', 0.0)
    n_range  = result['n_range']
    n_lo, n_hi = n_range
    n_modes_all = np.arange(n_lo, n_hi + 1)
    music_pseudo = result.get('music_pseudo')   # (n_win, n_freq, n_modes) or None

    thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

    has_n1rms = n1rms_dict is not None
    has_music = music_pseudo is not None

    n_panels = 4 + int(has_n1rms)
    if has_music:
        n_panels += 1
    ratios = [2.5, 1.5, 2, 2] + ([2] if has_music else []) + ([1] if has_n1rms else [])
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                              gridspec_kw={'height_ratios': ratios},
                              sharex=True)
    fig.subplots_adjust(hspace=0.06)
    ax_iter = iter(axes)

    def _vline(ax):
        if onset_ms is not None:
            ax.axvline(onset_ms, color='lime', ls='--', lw=0.9, alpha=0.8)

    # ── Power spectrogram ─────────────────────────────────────────────────────
    ax = next(ax_iter)
    pw_db = 10 * np.log10(pw.T + 1e-20)
    vmax  = np.percentile(pw_db, 99)
    im = ax.pcolormesh(t, f, pw_db, cmap='inferno',
                       vmin=vmax - 40, vmax=vmax, shading='nearest')
    ax.set_ylabel('Freq (kHz)')
    ax.set_ylim(result['f_min_khz'], result['f_max_khz'])
    plt.colorbar(im, ax=ax, pad=0.01).set_label('Power (dB)', fontsize=7)
    title = f'Shot {shot} — SVD Mode Analysis' if shot else 'SVD Mode Analysis'
    ax.set_title(title, fontsize=10)
    _vline(ax)

    # ── Coherence λ₁/Σλ ─────────────────────────────────────────────────────
    ax = next(ax_iter)
    im2 = ax.pcolormesh(t, f, coh.T, cmap='viridis', vmin=0, vmax=1, shading='nearest')
    ax.axhline(0, color='w', lw=0)   # dummy
    ax.set_ylabel('Freq (kHz)')
    ax.set_ylim(result['f_min_khz'], result['f_max_khz'])
    cb2 = plt.colorbar(im2, ax=ax, pad=0.01)
    cb2.set_label('λ₁/Σλ', fontsize=7)
    n_avg = result.get('n_avg', '?')
    ax.text(0.01, 0.97,
            f'c95 = {c95:.2f}  (n_avg={n_avg}, nsmooth={result.get("nsmooth",1)})',
            transform=ax.transAxes, fontsize=7, va='top', color='white')
    _vline(ax)

    # ── Dominant mode number ─────────────────────────────────────────────────
    ax = next(ax_iter)
    nd_masked = np.where(coh.T > thresh, nd.T.astype(float), np.nan)
    cmap_n = plt.get_cmap('RdBu_r', n_hi - n_lo + 1)
    im3 = ax.pcolormesh(t, f, nd_masked, cmap=cmap_n,
                         vmin=n_lo - 0.5, vmax=n_hi + 0.5, shading='nearest')
    ax.set_ylabel('Freq (kHz)')
    ax.set_ylim(result['f_min_khz'], result['f_max_khz'])
    cb3 = plt.colorbar(im3, ax=ax, pad=0.01)
    cb3.set_label('n', fontsize=7)
    cb3.set_ticks(n_modes_all)
    _vline(ax)

    # ── RMS per mode vs time ─────────────────────────────────────────────────
    ax = next(ax_iter)
    for n in n_modes_all:
        if n == 0:
            continue
        lw = 1.4 if abs(n) == 1 else 0.8
        ax.plot(t, rms.get(int(n), np.zeros_like(t)), color=_n_color(n),
                lw=lw, label=f'n={n}')
    ax.set_ylabel('RMS (G)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=6, ncol=4, framealpha=0.5)
    ax.grid(True, alpha=0.3, lw=0.5)
    _vline(ax)

    # ── MUSIC pseudospectrum vs time ─────────────────────────────────────────
    if has_music:
        ax = next(ax_iter)
        music_sum = music_pseudo.sum(axis=1)   # (n_win, n_modes)
        n_modes_list = list(np.arange(n_lo, n_hi + 1))
        for j, n in enumerate(n_modes_list):
            if n == 0:
                continue
            lw = 1.4 if abs(n) == 1 else 0.7
            ax.plot(t, music_sum[:, j], color=_n_color(n), lw=lw, label=f'n={n}')
        ax.set_ylabel('MUSIC (dB·kHz)', fontsize=9)
        ax.legend(loc='upper right', fontsize=6, ncol=4, framealpha=0.5)
        ax.grid(True, alpha=0.3, lw=0.5)
        _vline(ax)

    # ── N1RMS ────────────────────────────────────────────────────────────────
    if has_n1rms:
        ax = next(ax_iter)
        t_n = np.array(n1rms_dict.get('time', []))
        d_n = np.abs(np.array(n1rms_dict.get('data', [])))
        if t_n.size:
            m_n = (t_n >= t[0]) & (t_n <= t[-1])
            t_n, d_n = t_n[m_n], d_n[m_n]
        ax.plot(t_n, d_n, 'k', lw=0.7)
        ax.axhline(12, color='r', ls='--', lw=0.8)
        ax.set_ylabel('N1RMS (G)', fontsize=9)
        ax.set_ylim(0, max(15, d_n.max() * 1.1) if d_n.size else 20)
        ax.grid(True, alpha=0.3, lw=0.5)
        _vline(ax)

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    axes[-1].set_xlabel('Time (ms)')
    fig.align_ylabels(axes)

    # Colorbars on spectrogram panels shrink those axes horizontally; align all
    # panel right edges so the time axis is physically the same width everywhere.
    fig.canvas.draw()
    min_x1 = min(ax.get_position().x1 for ax in axes)
    for ax in axes:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0, min_x1 - p.x0, p.height])

    return fig


# ── Plotting ───────────────────────────────────────────────────────────────────

# Fixed per-n colors for line plots — distinct and visible on white background.
# n=1 red (primary NTM), n=2 blue, n=3 dark green (was invisible in RdBu_r),
# n=4 orange; negative-n in cooler/muted tones.
_N_COLORS = {
     1: '#d62728',   # red
     2: '#1f77b4',   # blue
     3: '#2ca02c',   # dark green
     4: '#ff7f0e',   # orange
    -1: '#e377c2',   # pink
    -2: '#17becf',   # cyan
    -3: '#8c564b',   # brown
     0: '#7f7f7f',   # grey
}
def _n_color(n):
    return _N_COLORS.get(int(n), plt.cm.tab10(abs(int(n)) % 10))

def plot_modespec(result, shot=None, n1rms_dict=None,
                  coh_thresh=None, onset_ms=None, figsize=(12, 10),
                  mode_label='n'):
    """
    Four-panel modespec-style plot.

    Panel 1: Power spectrogram [dB]
    Panel 2: Dominant mode number (masked by coherence > c95 or coh_thresh)
    Panel 3: RMS amplitude per mode vs time [G]
    Panel 4: N1RMS signal (optional)

    Parameters
    ----------
    result     : dict from mode_spectrogram
    shot       : int, for title
    n1rms_dict : dict with 'time' and 'data' keys, overlaid on panel 4
    coh_thresh : float override; defaults to result['c95'] (95% confidence)
    onset_ms   : float, draws vertical dashed line at TM onset
    mode_label : 'n' for toroidal (default) or 'm' for poloidal analysis
    """
    t   = result['t_win_ms']
    f   = result['freq_khz']
    pw  = result['power']
    nd  = result['n_dominant']
    coh = result['coherence']
    rms = result['rms_vs_time']
    n_range = result['n_range']
    c95     = result.get('c95', 0.0)

    thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

    n_lo, n_hi   = n_range
    n_modes_all  = np.arange(n_lo, n_hi + 1)

    n_panels = 4 if n1rms_dict is not None else 3
    ratios   = [3, 2, 2] + ([1] if n_panels == 4 else [])
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                              gridspec_kw={'height_ratios': ratios},
                              sharex=True)
    fig.subplots_adjust(hspace=0.06)

    def _vline(ax):
        if onset_ms is not None:
            ax.axvline(onset_ms, color='lime', ls='--', lw=0.9, alpha=0.8)

    # ── Panel 1: Power spectrogram ────────────────────────────────────────────
    ax = axes[0]
    pw_db = 10 * np.log10(pw.T + 1e-20)
    vmax  = np.percentile(pw_db, 99)
    im = ax.pcolormesh(t, f, pw_db, cmap='inferno',
                       vmin=vmax - 40, vmax=vmax, shading='nearest')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim(result['f_min_khz'], result['f_max_khz'])
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label('Power (dB)', fontsize=8)
    array_label = 'Toroidal' if mode_label == 'n' else 'Poloidal'
    title = f'Shot {shot} — {array_label} Mode Analysis' if shot else f'{array_label} Mode Analysis'
    ax.set_title(title, fontsize=10)
    _vline(ax)

    # ── Panel 2: Dominant mode number ─────────────────────────────────────────
    ax = axes[1]
    nd_masked = np.where(coh.T > thresh, nd.T.astype(float), np.nan)
    cmap_n = plt.get_cmap('RdBu_r', n_hi - n_lo + 1)
    im2 = ax.pcolormesh(t, f, nd_masked, cmap=cmap_n,
                         vmin=n_lo - 0.5, vmax=n_hi + 0.5, shading='nearest')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim(result['f_min_khz'], result['f_max_khz'])
    cb2 = plt.colorbar(im2, ax=ax, pad=0.01)
    cb2.set_label(mode_label, fontsize=8)
    cb2.set_ticks(n_modes_all)
    nsmooth = result.get('nsmooth', 1)
    ax.text(0.01, 0.97,
            f'coh > {thresh:.2f}  (c95={c95:.2f}, nsmooth={nsmooth})',
            transform=ax.transAxes, fontsize=7, va='top', color='white')
    _vline(ax)

    # ── Panel 3: RMS per mode vs time ─────────────────────────────────────────
    ax = axes[2]
    for n in n_modes_all:
        if n == 0:
            continue
        lw = 1.4 if abs(n) == 1 else 0.8
        ax.plot(t, rms.get(int(n), np.zeros_like(t)), color=_n_color(n),
                lw=lw, label=f'{mode_label}={n}')
    ax.set_ylabel('RMS (G)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=6, ncol=4, framealpha=0.5)
    ax.grid(True, alpha=0.3, lw=0.5)
    _vline(ax)

    # ── Panel 4 (optional): N1RMS ─────────────────────────────────────────────
    if n1rms_dict is not None:
        ax = axes[3]
        t_n = np.array(n1rms_dict.get('time', []))
        d_n = np.abs(np.array(n1rms_dict.get('data', [])))
        if t_n.size:
            m_n = (t_n >= t[0]) & (t_n <= t[-1])
            t_n, d_n = t_n[m_n], d_n[m_n]
        ax.plot(t_n, d_n, 'k', lw=0.7)
        ax.axhline(12, color='r', ls='--', lw=0.8, label='12 G')
        ax.set_ylabel('N1RMS (G)', fontsize=9)
        ax.set_ylim(0, max(15, d_n.max() * 1.1) if d_n.size else 20)
        ax.grid(True, alpha=0.3, lw=0.5)
        _vline(ax)

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    axes[-1].set_xlabel('Time (ms)')
    fig.align_ylabels(axes)

    # Colorbars on spectrogram panels shrink those axes horizontally; align all
    # panel right edges so the time axis is physically the same width everywhere.
    fig.canvas.draw()
    min_x1 = min(ax.get_position().x1 for ax in axes)
    for ax in axes:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0, min_x1 - p.x0, p.height])

    return fig


def plot_slice(slice_result, figsize=(11, 4)):
    """
    Two-panel plot for a single time-slice mode fit.

    Left:  probe amplitude and phase vs toroidal angle, with best-fit n line.
    Right: weighted chi² vs mode number n.
    """
    phi    = slice_result['phi_deg']
    power  = slice_result['power_vs_phi']
    phase  = np.degrees(slice_result['phase_vs_phi'])
    coh    = slice_result.get('coherence_vs_phi', np.ones_like(phi))
    n_modes= slice_result['n_modes']
    chi2   = slice_result['chi2_vs_n']
    n_best = slice_result['n_best']
    t0     = slice_result['t0_ms']
    f0     = slice_result['f0_khz']
    phi_fit    = slice_result.get('fit_curve_phi',   np.linspace(-50, 320, 200))
    phase_fit  = slice_result.get('fit_curve_phase', n_best * phi_fit % 360 - 180)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: amplitude bars + phase scatter + fit line
    ax1b = ax1.twinx()
    ax1.bar(phi, np.sqrt(power), width=4, alpha=0.5, color='C0', label='|B| (a.u.)')
    sc = ax1b.scatter(phi, phase, c=coh, cmap='viridis', s=40,
                       vmin=0, vmax=1, zorder=5, label='Phase (color=coh)')
    ax1b.plot(phi_fit, phase_fit, 'C3--', lw=1.2, label=f'n={n_best} fit')
    ax1b.set_ylim(-200, 200)
    ax1.set_xlabel('Toroidal angle (°)')
    ax1.set_ylabel('|B| (a.u.)', color='C0')
    ax1b.set_ylabel('Phase (°)', color='C1')
    ax1.set_title(f't = {t0:.0f} ms,  f = {f0:.1f} kHz')
    plt.colorbar(sc, ax=ax1b, label='Coherence', pad=0.12)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')

    # Right: weighted chi² vs n
    ax2.plot(n_modes, chi2, 'o-', lw=1.2)
    ax2.axvline(n_best, color='r', ls='--', lw=1, label=f'n={n_best} (best)')
    ax2.set_xlabel('Toroidal mode number n')
    ax2.set_ylabel('Weighted phase residual (rad²)')
    ax2.set_title('Mode number fit quality')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
