"""
Millisecond-resolution ECE zoom to check sawtooth sign inversion and
crash character (quiet vs active EPM phases).

Run: /fusion/projects/codes/conda/omega/envs_public/general/bin/python3 ece_ms_zoom.py
"""
import sys, os, json
sys.path.insert(0, '/home/yasodak/NTM_premptive_control')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHOTS = [199606, 199607]
LONG_EVENT = {199606: (2746, 4056), 199607: (3496, 4801)}

# ECE channels to plot (inner→outer): core is ECE46/42, q=2 ~ ECE22
CHANNELS_MS = [46, 42, 38, 34, 30, 26, 22]
LABELS_MS   = ['core','R~1.73','q_min','R~1.80','R~1.85','R~1.90','~q=2']

# Time windows: (tmin, tmax, label)  — 50–150 ms wide for crash visibility
WINDOWS = {
    199606: [
        (2600, 2760, 'quiet before onset'),
        (2750, 2900, 'at mode onset (2746ms)'),
        (3200, 3400, 'active window'),
        (4100, 4300, 'after long event'),
    ],
    199607: [
        (3200, 3400, 'quiet before onset'),
        (3450, 3600, 'at mode onset (3496ms)'),
        (3700, 3900, 'active window'),
        (4850, 5050, 'after long event'),
    ],
}

def load_ece(shot, ch):
    cp = f"/home/yasodak/exp/{shot}/ptdata_ECE{ch:02d}_{shot}.json"
    with open(cp) as f:
        d = json.load(f)
    return np.array(d['t']), np.array(d['y'])

def load_n1rms(shot):
    cp = f"/home/yasodak/exp/{shot}/mds_mhd_MHD__TOP_MIRNOV_N1RMS5.json"
    with open(cp) as f:
        d = json.load(f)
    t, y = np.array(d['t']), np.array(d['y'])
    if np.nanmedian(t) < 10:
        t = t * 1e3
    return t, y


for shot in SHOTS:
    print(f"\n=== Shot {shot} ===")
    ev = LONG_EVENT[shot]

    ece = {}
    for ch in CHANNELS_MS:
        try:
            t, y = load_ece(shot, ch)
            ece[ch] = (t, y)
        except:
            print(f"  ECE{ch} not cached")

    t_n1, n1 = load_n1rms(shot)
    wins = WINDOWS[shot]

    fig, axes = plt.subplots(len(CHANNELS_MS)+1, len(wins),
                              figsize=(4.5*len(wins), 2*(len(CHANNELS_MS)+1)),
                              gridspec_kw={'hspace': 0.04, 'wspace': 0.08})
    fig.suptitle(f'Shot {shot}: ECE ms-zoom (core→edge) — sawtooth check', fontsize=11)

    cmap = plt.cm.plasma
    for col, (tmin, tmax, title) in enumerate(wins):
        in_ev = ev[0] <= (tmin+tmax)/2 <= ev[1]

        # N1RMS row
        ax = axes[0, col]
        mk = (t_n1 >= tmin) & (t_n1 <= tmax)
        ax.plot(t_n1[mk], n1[mk], 'k', lw=0.8)
        ax.axhline(12, color='red', lw=0.7, ls='--')
        if in_ev:
            ax.set_facecolor('#fff0f0')
        ax.set_xlim(tmin, tmax)
        ax.set_title(title, fontsize=7, pad=2,
                     color='darkred' if in_ev else 'black')
        ax.tick_params(labelbottom=False, labelsize=6)
        if col == 0:
            ax.set_ylabel('N1RMS\n(G)', fontsize=6)

        for row, (ch, lbl) in enumerate(zip(CHANNELS_MS, LABELS_MS)):
            ax2 = axes[row+1, col]
            if ch in ece:
                t_e, y_e = ece[ch]
                mk = (t_e >= tmin) & (t_e <= tmax)
                color = cmap(row / len(CHANNELS_MS))
                ax2.plot(t_e[mk], y_e[mk], lw=0.7, color=color)
            if in_ev:
                ax2.set_facecolor('#fff0f0')
            ax2.set_xlim(tmin, tmax)
            ax2.tick_params(labelsize=5)
            if col == 0:
                ax2.set_ylabel(f'ECE{ch}\n{lbl}', fontsize=6)
            if row < len(CHANNELS_MS)-1:
                ax2.tick_params(labelbottom=False)
            else:
                ax2.set_xlabel('t (ms)', fontsize=6)

    out = f'figures/ece_ms_zoom_{shot}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"  Saved {out}")
    plt.close(fig)

    # ── Also: single-crash zoom — find a crash in the core channel and zoom 10ms ──
    # Detect crashes as rapid drops in ECE46 > 2*noise_level in <2ms
    if 46 in ece:
        t_c, y_c = ece[46]
        mask = (t_c >= 2200) & (t_c <= 5400)
        tc, yc = t_c[mask], y_c[mask]
        dy = np.diff(yc)
        noise = np.std(dy)
        # Crashes: large negative dy in one step
        crash_idx = np.where(dy < -5*noise)[0]
        crash_times = tc[crash_idx]
        print(f"  ECE46 crashes detected: {len(crash_times)}")
        if len(crash_times) > 0:
            print(f"  First 10: {crash_times[:10].astype(int).tolist()}")

        # Pick one crash in quiet phase and one in active phase
        quiet_crashes = crash_times[(crash_times < ev[0]) | (crash_times > ev[1])]
        active_crashes = crash_times[(crash_times >= ev[0]) & (crash_times <= ev[1])]

        sample_crashes = []
        if len(quiet_crashes) > 2:
            sample_crashes.append((quiet_crashes[len(quiet_crashes)//2], 'quiet phase'))
        if len(active_crashes) > 2:
            sample_crashes.append((active_crashes[len(active_crashes)//2], 'active phase'))

        if sample_crashes:
            fig3, axes3 = plt.subplots(len(CHANNELS_MS), len(sample_crashes),
                                        figsize=(5*len(sample_crashes), 2*len(CHANNELS_MS)),
                                        gridspec_kw={'hspace': 0.05, 'wspace': 0.1},
                                        sharex='col')
            fig3.suptitle(f'Shot {shot}: single crash zoom (±15 ms)', fontsize=11)

            for col, (tc_crash, phase_lbl) in enumerate(sample_crashes):
                for row, (ch, lbl) in enumerate(zip(CHANNELS_MS, LABELS_MS)):
                    ax3 = axes3[row, col] if len(sample_crashes) > 1 else axes3[row]
                    if ch in ece:
                        t_e, y_e = ece[ch]
                        mk = (t_e >= tc_crash-15) & (t_e <= tc_crash+15)
                        ax3.plot(t_e[mk], y_e[mk], lw=0.8,
                                 color=cmap(row / len(CHANNELS_MS)))
                    ax3.axvline(tc_crash, color='gray', lw=0.8, ls='--')
                    ax3.set_xlim(tc_crash-15, tc_crash+15)
                    ax3.tick_params(labelsize=6)
                    if row == 0:
                        ax3.set_title(f'{phase_lbl}\ncrash at {tc_crash:.1f} ms', fontsize=8)
                    if col == 0:
                        ax3.set_ylabel(f'ECE{ch} {lbl}', fontsize=6)
                    if row < len(CHANNELS_MS)-1:
                        ax3.tick_params(labelbottom=False)
                    else:
                        ax3.set_xlabel('t (ms)', fontsize=6)

            out3 = f'figures/ece_crash_zoom_{shot}.png'
            fig3.savefig(out3, dpi=150, bbox_inches='tight')
            print(f"  Saved {out3}")
            plt.close(fig3)

print("\nDone.")
