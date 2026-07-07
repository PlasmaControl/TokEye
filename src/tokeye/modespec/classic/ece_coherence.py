"""
ECE cross-channel coherence and phase — quiet vs active EPM phases.

Tests whether Te fluctuations present during 'quiet' phases are:
  (a) spatially coherent across radii → real sub-threshold mode
  (b) phase-locked between channels → confirms EPM spatial structure
  (c) change in phase relationship between quiet and active phases

ECE is at 5 kHz (0.2ms). EPM at 8-20 kHz aliases to:
  8 kHz → 2.0 kHz,  9 kHz → 1.0 kHz,  10 kHz → DC,
  11 kHz → 1.0 kHz, 12 kHz → 2.0 kHz, 20 kHz → DC

Run: /fusion/projects/codes/conda/omega/envs_public/general/bin/python3 ece_coherence.py
"""
import sys, os, json
sys.path.insert(0, '/home/yasodak/NTM_premptive_control')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import coherence, welch, csd

SHOTS = [199606, 199607]
LONG_EVENT = {199606: (2746, 4056), 199607: (3496, 4801)}

# Channels spanning core to edge; ECE22~q=2, ECE38~q_min, ECE46~core
CHANNELS = [22, 26, 30, 34, 38, 42, 46]
LABELS   = ['~q=2','R~1.90','R~1.85','R~1.80','~q_min','R~1.73','core']
REF_CH   = 38   # reference channel (q_min region — most active)

FS = 5000.0  # Hz (0.2ms cadence)
DT = 1.0/FS  # s

def load_ece(shot, ch):
    cp = f'/home/yasodak/exp/{shot}/ptdata_ECE{ch:02d}_{shot}.json'
    with open(cp) as f: d = json.load(f)
    t, y = np.array(d['t']), np.array(d['y'])
    if np.nanmedian(t) < 10: t = t*1e3
    return t, y

def load_n1rms(shot):
    cp = f'/home/yasodak/exp/{shot}/mds_mhd_MHD__TOP_MIRNOV_N1RMS5.json'
    with open(cp) as f: d = json.load(f)
    t, y = np.array(d['t']), np.array(d['y'])
    if np.nanmedian(t) < 10: t = t*1e3
    return t, y

def get_window(t, y, tmin, tmax):
    mk = (t >= tmin) & (t <= tmax) & np.isfinite(y)
    return y[mk]

# Segment definitions: quiet = pre-long-event, active = long event
SEGS = {
    # 199606: inter-burst quiet gaps within the burst period are < 40ms (< 200 pts at 5 kHz),
    #   too short for frequency-resolved coherence. Use pre-onset flatop as quiet reference.
    # 199607: quiet gap at ~3400ms is 115ms = 575 pts, insufficient for NPERSEG=1000.
    #   Use pre-onset (2200–3370ms) as the ECE quiet reference.
    #   The MPI analysis (200 kHz) uses the actual inter-burst gaps at 3400ms and 4800ms.
    199606: {
        'quiet':  (2000, 2740),   # pre-onset quiet (best available for ECE 5kHz)
        'active': (3200, 3800),   # burst period — N1RMS5 peaks 13–20 G, 600ms = 3000 pts
    },
    199607: {
        'quiet':  (2200, 3360),   # pre-onset (inter-burst gap at 3400ms too short for ECE)
        'active': (3800, 4700),   # main burst period (N1RMS5 12–28 G)
    },
}

# nperseg for Welch/coherence: 0.2s → 1000 pts (shorter segments = more averages)
NPERSEG = 1000

# Aliased EPM frequencies (EPM at 8–20 kHz aliases into ECE at 5 kHz)
EPM_ALIASES = {
    '1kHz (9/11 kHz)': (800, 1200),
    '2kHz (8/12/18 kHz)': (1700, 2200),
}
EPM_ALIAS_CENTERS = [1000, 2000]  # Hz — for vertical markers

for shot in SHOTS:
    print(f'\n=== Shot {shot} ===')
    ev = LONG_EVENT[shot]
    segs = SEGS[shot]

    # Load all ECE channels
    ece = {}
    t_ref = None
    for ch in CHANNELS:
        try:
            t, y = load_ece(shot, ch)
            ece[ch] = (t, y)
            if t_ref is None:
                t_ref = t
        except Exception as e:
            print(f'  ECE{ch}: not available')

    t_n1, n1 = load_n1rms(shot)

    # ── Figure: 3 columns = quiet / active / difference; rows = each channel pair ──
    fig, axes = plt.subplots(len(CHANNELS), 3, figsize=(15, 2.5*len(CHANNELS)),
                              gridspec_kw={'wspace': 0.3, 'hspace': 0.1})
    fig.suptitle(f'Shot {shot}: ECE cross-coherence with ECE{REF_CH} — quiet vs active',
                 fontsize=11)

    col_titles = [f"Quiet {segs['quiet']}ms", f"Active {segs['active']}ms",
                  'Coherence difference (active − quiet)']
    for c, ttl in enumerate(col_titles):
        axes[0, c].set_title(ttl, fontsize=9)

    for row, (ch, lbl) in enumerate(zip(CHANNELS, LABELS)):
        ax_q  = axes[row, 0]
        ax_a  = axes[row, 1]
        ax_d  = axes[row, 2]

        if ch not in ece or REF_CH not in ece:
            continue

        t_ch,  y_ch  = ece[ch]
        t_ref_ch, y_ref = ece[REF_CH]

        for ax, (tmin, tmax), color, phase_label in [
            (ax_q, segs['quiet'],  'tab:green',  'quiet'),
            (ax_a, segs['active'], 'tab:red',    'active'),
        ]:
            # extract segments on same time grid (both ECE channels same rate)
            mk = (t_ch >= tmin) & (t_ch <= tmax) & np.isfinite(y_ch)
            mk_r = (t_ref_ch >= tmin) & (t_ref_ch <= tmax) & np.isfinite(y_ref)
            n_pts = min(mk.sum(), mk_r.sum())
            if n_pts < NPERSEG * 2:
                ax.text(0.5, 0.5, 'insufficient data', transform=ax.transAxes,
                        ha='center', fontsize=7)
                continue
            y1 = y_ch[mk][:n_pts]
            y2 = y_ref[mk_r][:n_pts]

            # Cross-coherence
            f_c, coh = coherence(y1, y2, fs=FS, nperseg=NPERSEG)
            # Cross-spectral phase
            f_s, Pxy = csd(y1, y2, fs=FS, nperseg=NPERSEG)
            phase_xy = np.angle(Pxy, deg=True)

            ax.plot(f_c, coh, color=color, lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 2500)

            # Mark aliased EPM frequencies
            for f_epm in [8000, 9000, 10000, 11000, 12000, 15000, 18000, 20000]:
                f_alias = abs((f_epm % int(FS)) - (FS if (f_epm % int(FS)) > FS/2 else 0))
                if f_alias > 0:
                    ax.axvline(f_alias, color='gray', lw=0.4, ls=':', alpha=0.5)

            if row == len(CHANNELS)-1:
                ax.set_xlabel('f (Hz)', fontsize=6)
            ax.tick_params(labelsize=5)
            if ch == REF_CH:
                ax.set_ylim(0, 1)

        if row == 0 or True:
            axes[row, 0].set_ylabel(f'ECE{ch} {lbl}\nvs ECE{REF_CH}', fontsize=6)

        # Difference panel: coherence active - quiet
        try:
            tmin_q, tmax_q = segs['quiet']
            tmin_a, tmax_a = segs['active']
            mk_q = (t_ch >= tmin_q) & (t_ch <= tmax_q) & np.isfinite(y_ch)
            mk_a = (t_ch >= tmin_a) & (t_ch <= tmax_a) & np.isfinite(y_ch)
            mk_rq = (t_ref_ch >= tmin_q) & (t_ref_ch <= tmax_q) & np.isfinite(y_ref)
            mk_ra = (t_ref_ch >= tmin_a) & (t_ref_ch <= tmax_a) & np.isfinite(y_ref)
            n_q = min(mk_q.sum(), mk_rq.sum())
            n_a = min(mk_a.sum(), mk_ra.sum())
            if n_q >= NPERSEG*2 and n_a >= NPERSEG*2:
                _, coh_q = coherence(y_ch[mk_q][:n_q], y_ref[mk_rq][:n_q], fs=FS, nperseg=NPERSEG)
                f_a, coh_a = coherence(y_ch[mk_a][:n_a], y_ref[mk_ra][:n_a], fs=FS, nperseg=NPERSEG)
                diff = coh_a - coh_q
                ax_d.plot(f_a, diff, color='k', lw=0.8)
                ax_d.axhline(0, color='gray', lw=0.5)
                ax_d.fill_between(f_a, diff, 0,
                                   where=diff > 0, color='tab:red', alpha=0.4)
                ax_d.fill_between(f_a, diff, 0,
                                   where=diff < 0, color='tab:green', alpha=0.4)
                ax_d.set_ylim(-1, 1)
                ax_d.set_xlim(0, 2500)
                # Find frequency of max coherence increase
                if len(f_a) > 0:
                    peak = f_a[np.argmax(diff)]
                    ax_d.axvline(peak, color='tab:red', lw=0.8, ls='--')
                    print(f'  ECE{ch} vs ECE{REF_CH}: max coherence gain at {peak:.0f} Hz '
                          f'(Δcoh={np.max(diff):.2f})')
            if row == len(CHANNELS)-1:
                ax_d.set_xlabel('f (Hz)', fontsize=6)
            ax_d.tick_params(labelsize=5)
        except Exception as e:
            ax_d.text(0.5, 0.5, str(e)[:30], transform=ax_d.transAxes, fontsize=6)

    out = f'figures/ece_coherence_{shot}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  Saved {out}')
    plt.close(fig)

    # ── Summary figure: coherence at alias bands vs channel ──────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 7),
                                gridspec_kw={'hspace': 0.35, 'wspace': 0.3})
    fig2.suptitle(f'Shot {shot}: ECE coherence at EPM alias frequencies vs channel\n'
                  f'(ref = ECE{REF_CH}, quiet={segs["quiet"]}ms, active={segs["active"]}ms)',
                  fontsize=10)

    # For each alias band collect coherence per channel
    alias_bands = [(800, 1200, '~1 kHz\n(9/11 kHz EPM alias)'),
                   (1700, 2300, '~2 kHz\n(8/12/18 kHz EPM alias)')]

    # Also compute full-band max coherence per channel
    coh_summary = {ch: {'quiet': {}, 'active': {}} for ch in CHANNELS}

    for ch in CHANNELS:
        if ch not in ece or REF_CH not in ece:
            continue
        t_ch, y_ch = ece[ch]
        t_ref_ch, y_ref = ece[REF_CH]
        for phase_key, (tmin, tmax) in [('quiet', segs['quiet']), ('active', segs['active'])]:
            mk = (t_ch >= tmin) & (t_ch <= tmax) & np.isfinite(y_ch)
            mk_r = (t_ref_ch >= tmin) & (t_ref_ch <= tmax) & np.isfinite(y_ref)
            n_pts = min(mk.sum(), mk_r.sum())
            if n_pts < NPERSEG * 2:
                continue
            f_c, coh = coherence(y_ch[mk][:n_pts], y_ref[mk_r][:n_pts], fs=FS, nperseg=NPERSEG)
            for (flo, fhi, _) in alias_bands:
                band = (f_c >= flo) & (f_c <= fhi)
                coh_summary[ch][phase_key][(flo, fhi)] = np.mean(coh[band]) if band.sum() > 0 else np.nan
            # Broadband 50–2000 Hz
            band_bb = (f_c >= 50) & (f_c <= 2000)
            coh_summary[ch][phase_key]['broad'] = np.mean(coh[band_bb]) if band_bb.sum() > 0 else np.nan

    x_pos = np.arange(len(CHANNELS))
    bar_w = 0.35

    for ai, (flo, fhi, band_lbl) in enumerate(alias_bands):
        ax_b = axes2[ai, 0]
        ax_d2 = axes2[ai, 1]

        coh_q_arr = [coh_summary[ch]['quiet'].get((flo, fhi), np.nan) for ch in CHANNELS]
        coh_a_arr = [coh_summary[ch]['active'].get((flo, fhi), np.nan) for ch in CHANNELS]

        ax_b.bar(x_pos - bar_w/2, coh_q_arr, bar_w, color='tab:green', alpha=0.8,
                 label=f'Quiet {segs["quiet"]}ms')
        ax_b.bar(x_pos + bar_w/2, coh_a_arr, bar_w, color='tab:red', alpha=0.8,
                 label=f'Active {segs["active"]}ms')
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([f'ECE{c}\n{l}' for c, l in zip(CHANNELS, LABELS)],
                              fontsize=6, rotation=30)
        ax_b.set_ylim(0, 1)
        ax_b.set_ylabel('Mean coherence', fontsize=8)
        ax_b.set_title(band_lbl, fontsize=8)
        ax_b.legend(fontsize=7)
        ax_b.axhline(2/NPERSEG * np.log(20), color='k', lw=0.8, ls='--', label='95% sig.')
        ax_b.tick_params(labelsize=6)

        diff_arr = np.array(coh_a_arr) - np.array(coh_q_arr)
        ax_d2.bar(x_pos, diff_arr,
                  color=['tab:red' if d > 0 else 'tab:green' for d in diff_arr])
        ax_d2.axhline(0, color='k', lw=0.8)
        ax_d2.set_xticks(x_pos)
        ax_d2.set_xticklabels([f'ECE{c}' for c in CHANNELS], fontsize=7, rotation=30)
        ax_d2.set_ylim(-1, 1)
        ax_d2.set_ylabel('Δcoh (active − quiet)', fontsize=8)
        ax_d2.set_title(f'Coherence gain {band_lbl}', fontsize=8)
        ax_d2.tick_params(labelsize=6)

        # Print summary
        print(f'  [{band_lbl.split(chr(10))[0]}] coherence per channel:')
        for ch, cq, ca in zip(CHANNELS, coh_q_arr, coh_a_arr):
            dc = ca - cq if not (np.isnan(ca) or np.isnan(cq)) else np.nan
            sq = f'{cq:.3f}' if not np.isnan(cq) else '---'
            sa = f'{ca:.3f}' if not np.isnan(ca) else '---'
            sd = f'{dc:.3f}' if not np.isnan(dc) else '---'
            print(f'    ECE{ch}: quiet={sq}, active={sa}, Δ={sd}')

    out2 = f'figures/ece_coherence_summary_{shot}.png'
    fig2.savefig(out2, dpi=130, bbox_inches='tight')
    print(f'  Saved {out2}')
    plt.close(fig2)

print('\nDone.')
