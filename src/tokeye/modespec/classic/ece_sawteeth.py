"""
ECE Te time traces for sawtooth/quasi-sawtooth check.
Channels ECE10-48 via PTDATA, 0.2ms cadence.

B0 ~ 2.0T at R0=1.7m → f_2nd_harmonic = 2*28*B GHz
  - Core (R~1.70m, B~2.0T):   f ~ 112 GHz → channel ~39
  - q_min (R~1.75m, B~1.94T): f ~ 109 GHz → channel ~36
  - q=2   (R~2.00m, B~1.70T): f ~  95 GHz → channel ~22
  - Edge  (R~2.25m, B~1.51T): f ~  85 GHz → channel ~12

Run: /fusion/projects/codes/conda/omega/envs_public/general/bin/python3 ece_sawteeth.py
"""
import sys, os, json, subprocess
sys.path.insert(0, '/home/yasodak/NTM_premptive_control')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHOTS = [199606, 199607]
# Channels chosen to span core → edge
# (higher channel number ~ higher freq ~ smaller R ~ hotter core)
CHANNELS = [12, 18, 22, 26, 30, 34, 38, 42, 46]
LABELS   = ['~edge','R~2.1m','~q=2','R~1.9m','R~1.85m',
            'R~1.80m','~q_min','R~1.73m','core']
LONG_EVENT = {199606: (2746, 4056), 199607: (3496, 4801)}

# Zoom windows: (wide_tmin, wide_tmax, tight_tmin, tight_tmax)
ZOOMS = {
    199606: [(2600, 2900, 2700, 2800)],   # around onset at 2746ms
    199607: [(3300, 3650, 3450, 3560)],   # around onset at 3496ms
}

def fetch_ece(shot, ch):
    key = f"ptdata_ECE{ch:02d}_{shot}"
    cp  = f"/home/yasodak/exp/{shot}/{key}.json"
    os.makedirs(f"/home/yasodak/exp/{shot}", exist_ok=True)
    if os.path.exists(cp):
        with open(cp) as f:
            d = json.load(f)
        return np.array(d['t']), np.array(d['y'])
    cmd = (f"python3 -c \""
           f"import MDSplus as mds; c=mds.Connection('atlas.gat.com');"
           f"c.openTree('d3d',{shot});"
           f"s=c.get('ptdata2(\\\"ECE{ch:02d}\\\",{shot})');"
           f"t=c.get('dim_of(ptdata2(\\\"ECE{ch:02d}\\\",{shot}))');"
           f"import json; print(json.dumps({{'t':list(map(float,t)),'y':list(map(float,s))}}))\"")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    d = json.loads(r.stdout.strip())
    with open(cp, 'w') as f:
        json.dump(d, f)
    return np.array(d['t']), np.array(d['y'])

def fetch_n1rms(shot):
    cp = f"/home/yasodak/exp/{shot}/mds_mhd_MHD__TOP_MIRNOV_N1RMS5.json"
    with open(cp) as f:
        d = json.load(f)
    t, y = np.array(d['t']), np.array(d['y'])
    if np.nanmedian(t) < 10:
        t = t * 1e3
    return t, y

# ── Figure 1: full flat-top overview for each shot ────────────────────────────
for shot in SHOTS:
    print(f"\nShot {shot}")
    ev = LONG_EVENT[shot]

    ece_data = {}
    for ch in CHANNELS:
        t, y = fetch_ece(shot, ch)
        if t is not None:
            ece_data[ch] = (t, y)
            print(f"  ECE{ch:02d}: {len(t)} pts")

    t_n1, n1 = fetch_n1rms(shot)

    fig, axes = plt.subplots(len(CHANNELS)+1, 1, figsize=(14, 2*(len(CHANNELS)+1)),
                              sharex=True, gridspec_kw={'hspace': 0.05})
    fig.suptitle(f'Shot {shot}: ECE channels (core→edge) vs N1RMS', fontsize=11)

    # N1RMS top
    ax = axes[0]
    mk = (t_n1 >= 2000) & (t_n1 <= 5500)
    ax.plot(t_n1[mk], n1[mk], 'k', lw=0.7)
    ax.axhline(12, color='red', lw=0.7, ls='--')
    ax.axvspan(*ev, color='tab:red', alpha=0.1)
    ax.set_ylabel('N1RMS\n(G)', fontsize=7)

    cmap = plt.cm.plasma
    for i, (ch, lbl) in enumerate(zip(CHANNELS[::-1], LABELS[::-1])):  # inner→outer
        ax = axes[i+1]
        if ch in ece_data:
            t, y = ece_data[ch]
            mk = (t >= 2000) & (t <= 5500)
            # downsample to 1ms for overview
            step = max(1, int(5 / np.median(np.diff(t[mk]))))
            ax.plot(t[mk][::step], y[mk][::step], lw=0.5,
                    color=cmap(i / len(CHANNELS)))
        ax.axvspan(*ev, color='tab:red', alpha=0.1)
        ax.set_ylabel(f'ECE{ch}\n{lbl}', fontsize=6)
        ax.tick_params(labelsize=6)

    axes[-1].set_xlabel('Time (ms)', fontsize=8)
    axes[-1].set_xlim(2000, 5500)

    out = f'figures/ece_overview_{shot}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"  Saved {out}")
    plt.close(fig)

    # ── Figure 2: zoom windows ────────────────────────────────────────────────
    for zi, (tw0, tw1, tt0, tt1) in enumerate(ZOOMS[shot]):
        fig2, axes2 = plt.subplots(len(CHANNELS)+1, 1,
                                    figsize=(12, 2*(len(CHANNELS)+1)),
                                    sharex=True, gridspec_kw={'hspace': 0.04})
        fig2.suptitle(f'Shot {shot}: ECE zoom {tw0}–{tw1} ms  '
                      f'(mode onset {ev[0]} ms, yellow = tight window)', fontsize=10)

        ax = axes2[0]
        mk = (t_n1 >= tw0) & (t_n1 <= tw1)
        ax.plot(t_n1[mk], n1[mk], 'k', lw=0.8)
        ax.axhline(12, color='red', lw=0.8, ls='--')
        ax.axvspan(*ev, color='tab:red', alpha=0.15)
        ax.axvspan(tt0, tt1, color='yellow', alpha=0.25)
        ax.set_ylabel('N1RMS\n(G)', fontsize=7)

        for i, (ch, lbl) in enumerate(zip(CHANNELS[::-1], LABELS[::-1])):
            ax2 = axes2[i+1]
            if ch in ece_data:
                t_e, y_e = ece_data[ch]
                mk = (t_e >= tw0) & (t_e <= tw1)
                ax2.plot(t_e[mk], y_e[mk], lw=0.6,
                         color=cmap(i / len(CHANNELS)))
                mk2 = (t_e >= tt0) & (t_e <= tt1)
                ax2.plot(t_e[mk2], y_e[mk2], lw=1.5,
                         color=cmap(i / len(CHANNELS)))
            ax2.axvspan(*ev, color='tab:red', alpha=0.12)
            ax2.axvspan(tt0, tt1, color='yellow', alpha=0.2)
            ax2.set_ylabel(f'ECE{ch}\n{lbl}', fontsize=6)
            ax2.tick_params(labelsize=6)

        axes2[-1].set_xlabel('Time (ms)', fontsize=8)
        axes2[-1].set_xlim(tw0, tw1)

        out2 = f'figures/ece_zoom_{shot}_w{zi}.png'
        fig2.savefig(out2, dpi=130, bbox_inches='tight')
        print(f"  Saved {out2}")
        plt.close(fig2)

print("\nDone.")
