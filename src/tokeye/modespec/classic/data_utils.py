"""
data_utils.py — MDSplus fetch helpers, pkl loader, and analysis utilities.

Split out from analysis.ipynb so every function is importable and testable
without opening a notebook.
"""

import os, pickle
import numpy as np
from scipy import signal as scipy_signal

try:
    import MDSplus as mds
    MDS_AVAILABLE = True
except ImportError:
    MDS_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────

ATLAS   = 'atlas.gat.com'
PKL_DIR = '/fusion/projects/xpsi/transient_control/rothsteina/MRE/data'

# TM survival model — best model: rt_model_latest_pcb (38 features, valid_loss 0.1855)
# model_23_2 (231 feat, valid_loss 0.4277): hyperparameter trial only, no companion norm files
# model_25   (42 feat, valid_loss 0.5003):  use dsm_infer_cakenn.py + cake_normalizations_dict.pkl
_HERE = os.path.dirname(os.path.abspath(__file__))
TM_MODEL_PKL = os.path.join(_HERE, 'rt_model_latest_pcb.pkl')       # 38-feature PCB ensemble (best)
TM_NORM_PKL  = os.path.join(_HERE, 'rt_normalizations_bms_pcb.pkl') # scalar + PCA score norms
TM_PCA_PKL   = os.path.join(_HERE, 'rt_pca_components_bms_pcb.pkl') # 6-profile PCA components

# model_25: CAKENN-format 42-feature model (valid_loss 0.5003, for comparison)
TM_CAKE_MODEL_PKL = os.path.join(_HERE, 'model_25.pkl')
TM_CAKE_NORM_PKL  = os.path.join(_HERE, 'cake_normalizations_dict.pkl')
TM_CAKE_PCA_PKL   = os.path.join(_HERE, 'cakenn_pca_components.pkl')
DSM_INFER_CAKE_SCRIPT = os.path.join(_HERE, 'dsm_infer_cakenn.py')

# TAPE C model (k2c weights, what PCS actually ran during shots)
# Note: tape_rtprofile weights differ from rt_model_latest_pcb.pkl — separately trained
TAPE_C_INFER_SCRIPT = os.path.join(_HERE, 'tape_c_infer.py')
TAPE_OFFLINE_BIN    = os.path.join(_HERE, 'tape_offline')

# Radial grid used by the model (33 uniform rho_N points, same as training)
TM_PROFILE_GRID = np.linspace(0, 1, 33)

# ── Signal definitions ─────────────────────────────────────────────────────────

GYROTRONS = {'LEIA': 4, 'R2D2': 5, 'YODA': 8, 'NASA': 9, 'HAN': 11}

N1RMS_CANDIDATES = ['N1RMS', 'RMS01', 'N01RMS', 'BRMSA']

MPI_322_SIGNALS = [
    'MPI1A322D', 'MPI2A322D', 'MPI3A322D', 'MPI4A322D', 'MPI5A322D',
    'MPI11M322D',
    'MPI1B322D', 'MPI2B322D', 'MPI3B322D', 'MPI4B322D', 'MPI5B322D',
]

PTDATA_OVERVIEW = {
    'ip': 'IP',   # A
    # echpwr is fetched via RF tree (not PTDATA — PTDATA ECHPWR units are unknown)
}

TREE_OVERVIEW = {
    'EFIT02ER': {
        'wmhd':      '\\EFIT02ER::TOP.RESULTS.AEQDSK:WMHD',     # J
        'betan_efit':'\\EFIT02ER::TOP.RESULTS.AEQDSK:BETAN',
        'aminor':    '\\EFIT02ER::TOP.RESULTS.AEQDSK:AMINOR',   # m
        'cpasma':    '\\EFIT02ER::TOP.RESULTS.GEQDSK:CPASMA',   # A
        'rmaxis':    '\\EFIT02ER::TOP.RESULTS.GEQDSK:RMAXIS',   # m (AEQDSK:RMAXIS not in EFIT02ER)
        'betap':     '\\EFIT02ER::TOP.RESULTS.AEQDSK:BETAP',
        'ipmhd':     '\\EFIT02ER::TOP.RESULTS.AEQDSK:IPMHD',   # A
        # Extra scalars needed by TM model (post-shot equivalents of EFITRT2 inputs)
        'qmin':      '\\EFIT02ER::TOP.RESULTS.AEQDSK:QMIN',
        'li':        '\\EFIT02ER::TOP.RESULTS.AEQDSK:LI',
        'kappa':     '\\EFIT02ER::TOP.RESULTS.AEQDSK:KAPPA',
        'tribot':    '\\EFIT02ER::TOP.RESULTS.AEQDSK:TRIBOT',
        'tritop':    '\\EFIT02ER::TOP.RESULTS.AEQDSK:TRITOP',
        'volume':    '\\EFIT02ER::TOP.RESULTS.AEQDSK:VOLUME',   # m³
    },
    'BOLOM':     {'prad': '\\BOLOM::TOP.PRAD_01.PRAD:PRAD_TOT'},   # W
    'TRANSPORT': {'h98':  '\\TRANSPORT::TOP.AOT.TAU:H98Y2_OT'},
    'NB':        {'pinj': '\\NB::TOP:PINJ',                        # kW
                  'tinj': '\\NB::TOP:TINJ'},                       # N·m (NBI torque)
}

# ── Pickle cache helpers ───────────────────────────────────────────────────────

def _shot_dir(shot, data_dir):
    d = os.path.join(data_dir, str(shot))
    os.makedirs(d, exist_ok=True)
    return d

def _cache_path(shot, key, data_dir):
    return os.path.join(_shot_dir(shot, data_dir), f'{shot}_{key}.pkl')

def save_cache(shot, key, data, data_dir):
    with open(_cache_path(shot, key, data_dir), 'wb') as f:
        pickle.dump(data, f)

def load_cache(shot, key, data_dir):
    p = _cache_path(shot, key, data_dir)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    # Backward-compat: read old JSON cache if pkl not yet written
    p_json = os.path.join(_shot_dir(shot, data_dir), f'{shot}_{key}.json')
    if os.path.exists(p_json):
        import json
        with open(p_json) as f:
            return json.load(f)
    return None

def fetch_or_load(shot, key, fetch_fn, data_dir):
    """Return cached pkl if present (falls back to JSON); otherwise fetch, cache as pkl, return."""
    cached = load_cache(shot, key, data_dir)
    if cached is not None:
        print(f'  [{shot}] loaded "{key}" from cache')
        return cached
    if not MDS_AVAILABLE:
        raise RuntimeError(
            f'MDSplus unavailable and no cache for {shot}/{key}. MDSplus '
            'ships on the GA cluster and on conda-forge (conda install -c '
            'conda-forge mdsplus); fetching also needs atlas.gat.com access.'
        )
    print(f'  [{shot}] fetching "{key}" ...')
    result = fetch_fn()
    save_cache(shot, key, result, data_dir)
    return result

# ── Low-level MDSplus ──────────────────────────────────────────────────────────

def fetch_ptdata(shot, signal):
    """Fetch a PTDATA time-series. Returns (data_array, time_ms_array)."""
    conn = mds.Connection(ATLAS)
    conn.openTree('D3D', shot)
    data = np.array(conn.get(f'PTDATA("{signal}", {shot})').data())
    time = np.array(conn.get(f'DIM_OF(PTDATA("{signal}", {shot}))').data())
    conn.closeAllTrees()
    if time.size > 0 and np.max(np.abs(time)) < 100:
        time = time * 1e3   # s → ms
    return data, time

def fetch_tree_nodes(shot, tree, nodes_dict):
    """Fetch multiple nodes from one tree. Returns {key: {data, time}}."""
    results = {}
    conn = mds.Connection(ATLAS)
    try:
        conn.openTree(tree, shot)
        for key, node in nodes_dict.items():
            try:
                d = np.array(conn.get(node).data())
                t = np.array(conn.get(f'DIM_OF({node})').data())
                results[key] = {'data': d.tolist(), 'time': t.tolist()}
            except Exception as e:
                print(f'  Warning {key} ({tree}): {e}')
        conn.closeAllTrees()
    except Exception as e:
        print(f'  Could not open {tree} for {shot}: {e}')
    return results

def ts(d, key):
    """Unpack (time_array, data_array) from a {data, time} signals dict."""
    arr = d.get(key, {})
    if not arr:
        return np.array([]), np.array([])
    t = np.array(arr.get('time', arr.get('time_basis', [])))
    v = np.array(arr.get('data', arr.get('time_series', [])))
    return t, v

# ── Signal-set fetchers ────────────────────────────────────────────────────────

def fetch_ech_power(shot):
    """
    Fetch total ECH power in MW.

    Primary source: RF tree, where all FPWRC nodes are in Watts.
      \\RF::TOP.ECH.TOTAL:ECHPWRC          — total (preferred)
      \\RF::TOP.ECH.{GYRO}:EC{ABBREV}FPWRC — per gyrotron, summed as fallback

    PTDATA 'ECHPWR' is NOT used — its units are unknown (not MW, not kW).

    Returns
    -------
    dict with keys 'data' (MW), 'time' (ms), 'signal' (source label).
    """
    try:
        conn = mds.Connection(ATLAS)
        conn.openTree('RF', shot)

        # Try total node first
        total_node = '\\RF::TOP.ECH.TOTAL:ECHPWRC'
        try:
            d = np.array(conn.get(total_node).data())
            t = np.array(conn.get(f'DIM_OF({total_node})').data())
            if d.size > 1 and d.max() > 0:
                if t.size > 0 and np.max(np.abs(t)) < 100:
                    t = t * 1e3
                conn.closeAllTrees()
                print(f'  ECH: RF TOTAL:ECHPWRC, peak={d.max()/1e6:.3f} MW')
                return {'signal': total_node, 'data': (d / 1e6).tolist(), 'time': t.tolist()}
        except Exception:
            pass

        # Fallback: sum per-gyrotron FPWRC nodes
        time_ref = None
        total_pwr = None
        for gyro, _ in GYROTRONS.items():
            abbrev = gyro[:3].upper()
            node = f'\\RF::TOP.ECH.{gyro}:EC{abbrev}FPWRC'
            try:
                d = np.array(conn.get(node).data())
                t = np.array(conn.get(f'DIM_OF({node})').data())
                if d.size < 2 or d.max() == 0:
                    continue
                if t.size > 0 and np.max(np.abs(t)) < 100:
                    t = t * 1e3
                pwr_mw = d / 1e6
                if time_ref is None:
                    time_ref = t
                    total_pwr = pwr_mw.copy()
                else:
                    total_pwr += np.interp(time_ref, t, pwr_mw, left=0.0, right=0.0)
            except Exception:
                pass

        conn.closeAllTrees()

        if total_pwr is not None:
            print(f'  ECH: sum of FPWRC gyrotrons, peak={total_pwr.max():.3f} MW')
            return {'signal': 'RF_FPWRC_sum', 'data': total_pwr.tolist(), 'time': time_ref.tolist()}

    except Exception as e:
        print(f'  ECH RF tree failed: {e}')

    print('  Warning: ECH power not found in RF tree — no ECHPWR stored')
    return {'signal': None, 'data': [], 'time': []}


def fetch_overview(shot):
    results = {}
    for key, sig in PTDATA_OVERVIEW.items():
        try:
            d, t = fetch_ptdata(shot, sig)
            results[key] = {'data': d.tolist(), 'time': t.tolist()}
        except Exception as e:
            print(f'  Warning {key}: {e}')
    results['echpwr'] = fetch_ech_power(shot)
    for tree, nodes in TREE_OVERVIEW.items():
        results.update(fetch_tree_nodes(shot, tree, nodes))
    return results

def fetch_n1rms(shot):
    # Primary: MHD tree — use 5 ms smoothed version; fall back to full-rate
    for node, label in [
        ('\\MHD::TOP.MIRNOV:N1RMS5', 'MHD:N1RMS5'),
        ('\\MHD::TOP.MIRNOV:N1RMS',  'MHD:N1RMS'),
    ]:
        try:
            conn = mds.Connection(ATLAS)
            conn.openTree('MHD', shot)
            d = np.array(conn.get(node).data())
            t = np.array(conn.get(f'DIM_OF({node})').data())
            conn.closeAllTrees()
            if t.size > 0 and np.max(np.abs(t)) < 100:
                t = t * 1e3
            if d.size > 0:
                print(f'  Using: {label}')
                return {'signal': label, 'data': d.tolist(), 'time': t.tolist()}
        except Exception:
            continue
    # Fallback: PTDATA candidates
    for sig in N1RMS_CANDIDATES:
        try:
            d, t = fetch_ptdata(shot, sig)
            if d.size > 0:
                print(f'  Using: {sig}')
                return {'signal': sig, 'data': d.tolist(), 'time': t.tolist()}
        except Exception:
            continue
    print(f'  Warning: no N1RMS signal found for {shot}')
    return {'signal': None, 'data': [], 'time': []}

def fetch_n2rms(shot):
    for node, label in [
        ('\\MHD::TOP.MIRNOV:N2RMS5', 'MHD:N2RMS5'),
        ('\\MHD::TOP.MIRNOV:N2RMS',  'MHD:N2RMS'),
    ]:
        try:
            conn = mds.Connection(ATLAS)
            conn.openTree('MHD', shot)
            d = np.array(conn.get(node).data())
            t = np.array(conn.get(f'DIM_OF({node})').data())
            conn.closeAllTrees()
            if t.size > 0 and np.max(np.abs(t)) < 100:
                t = t * 1e3
            if d.size > 0:
                print(f'  Using: {label}')
                return {'signal': label, 'data': d.tolist(), 'time': t.tolist()}
        except Exception:
            continue
    for sig in ['N2RMS', 'RMS02', 'N02RMS']:
        try:
            d, t = fetch_ptdata(shot, sig)
            if d.size > 0:
                print(f'  Using: {sig}')
                return {'signal': sig, 'data': d.tolist(), 'time': t.tolist()}
        except Exception:
            continue
    print(f'  Warning: no N2RMS signal found for {shot}')
    return {'signal': None, 'data': [], 'time': []}

def fetch_neutron_rate(shot):
    """
    Fetch total neutron rate from IONS tree.

    Source: \\IONS::NEUTRONSRATE — 50 kHz, full shot, units n/s.
    The IONS tree is accessible from atlas.gat.com.

    Returns
    -------
    dict with keys 'signal', 'data' (n/s), 'time' (ms).
    """
    node = '\\IONS::NEUTRONSRATE'
    try:
        conn = mds.Connection(ATLAS)
        conn.openTree('IONS', shot)
        d = np.asarray(conn.get(f'float({node})'), dtype=float)
        t = np.asarray(conn.get(f'float(dim_of({node}))'), dtype=float)
        conn.closeAllTrees()
        if t.size > 0 and np.max(np.abs(t)) < 100:
            t = t * 1e3
        if d.size > 1:
            print(f'  [{shot}] NEUTRONSRATE: {d.size} pts, '
                  f't=[{t[0]:.0f},{t[-1]:.0f}] ms, '
                  f'fs={1e3/np.median(np.diff(t)):.0f} Hz')
            return {'signal': node, 'data': d.tolist(), 'time': t.tolist()}
    except Exception as e:
        print(f'  Warning fetch_neutron_rate({shot}): {e}')
    return {'signal': None, 'data': [], 'time': []}


def fetch_mirror_angles(shot):
    """
    Fetch poloidal and toroidal mirror angles for each ECH gyrotron.

    Primary: RF tree  \\RF::TOP.ECH.{GYRO}:EC{ABBREV}POLANG / AZIANG
    Fallback: PTDATA  GYSMPOL{N} / GYSMAZI{N}

    Keys produced: {GYRO}_pol_meas, {GYRO}_tor_meas
    """
    results = {}

    # --- RF tree (primary) ---
    try:
        conn = mds.Connection(ATLAS)
        conn.openTree('RF', shot)
        for gyro in GYROTRONS:
            abbrev = gyro[:3].upper()
            for sfx, rfsuffix in [('pol_meas', f'EC{abbrev}POLANG'),
                                   ('tor_meas', f'EC{abbrev}AZIANG')]:
                node = f'\\RF::TOP.ECH.{gyro}:{rfsuffix}'
                try:
                    d = np.array(conn.get(node).data())
                    t = np.array(conn.get(f'DIM_OF({node})').data())
                    if t.size > 0 and np.max(np.abs(t)) < 100:
                        t = t * 1e3
                    results[f'{gyro}_{sfx}'] = {'data': d.tolist(), 'time': t.tolist()}
                    print(f'  {gyro} {sfx}: {d.size} pts via RF tree')
                except Exception as e:
                    print(f'  Warning RF {gyro} {rfsuffix}: {e}')
        conn.closeAllTrees()
    except Exception as e:
        print(f'  RF tree unavailable: {e}')

    # --- PTDATA fallback for any missing signals ---
    for gyro, idx in GYROTRONS.items():
        for sfx, sig in [('pol_meas', f'GYSMPOL{idx}'),
                         ('tor_meas', f'GYSMAZI{idx}')]:
            key = f'{gyro}_{sfx}'
            if key in results:
                continue
            try:
                d, t = fetch_ptdata(shot, sig)
                results[key] = {'data': d.tolist(), 'time': t.tolist()}
                print(f'  {gyro} {sfx}: {d.size} pts via PTDATA {sig}')
            except Exception as e:
                print(f'  Warning PTDATA {sig}: {e}')
    return results

def fetch_mpi(shot, signals=None):
    if signals is None:
        signals = MPI_322_SIGNALS
    results = {}
    for sig in signals:
        try:
            d, t = fetch_ptdata(shot, sig)
            results[sig] = {'data': d.tolist(), 'time': t.tolist()}
        except Exception as e:
            print(f'  Warning {sig}: {e}')
    return results

# Conda Python 3.11 with torch + auton-survival available
DSM_CONDA_PY     = '/fusion/projects/codes/conda/omega/envs_public/general/bin/python3'
DSM_INFER_SCRIPT = os.path.join(_HERE, 'dsm_infer_38.py')


def fetch_thomson_cer(shot):
    """
    Fetch Te, ne, Ti, and rotation profiles from ELECTRONS/IONS trees.

    Signal priority (from common.py):
      Te, ne : ELECTRONS ZIPFIT → TS blessed core → TS r00
      Ti     : IONS ZIPFIT:ITEMPFIT (already keV)
      rot    : IONS ZIPFIT_CERA:TROTFIT → ZIPFIT:TROTFIT  (km/s)

    CER tree has NOPATH on atlas — all CER-derived data fetched from IONS tree.

    Returns dict {key: {data, rho, time}} where:
      data shape : (n_rho, n_t)  — indexed [rho_idx, time_idx]
      rho        : 1-D normalised grid (0–1)
      time       : 1-D time axis [ms]
    """
    if not MDS_AVAILABLE:
        return {}
    results = {}

    # ── Electron profiles (ELECTRONS tree) ──────────────────────────────────
    elec_candidates = {
        'Te_keV': [
            ('\\ELECTRONS::TOP.PROFILE_FITS.ZIPFIT:ETEMPFIT',     'ELECTRONS', 1.0),   # keV
            ('\\ELECTRONS::TOP.TS.BLESSED.CORE.FITTED:TE',        'ELECTRONS', 1e-3),  # eV
            ('\\ELECTRONS::TOP.TS.REVISIONS.REVISION00.CORE:TE',  'ELECTRONS', 1e-3),  # eV
        ],
        'ne': [
            ('\\ELECTRONS::TOP.PROFILE_FITS.ZIPFIT:EDENSFIT',    'ELECTRONS', 1.0),
            ('\\ELECTRONS::TOP.TS.BLESSED.CORE.FITTED:NE',        'ELECTRONS', 1.0),
            ('\\ELECTRONS::TOP.TS.REVISIONS.REVISION00.CORE:NE',  'ELECTRONS', 1.0),
        ],
    }
    for key, candidates in elec_candidates.items():
        for node, tree, scale in candidates:
            try:
                conn = mds.Connection(ATLAS)
                conn.openTree(tree, shot)
                d = np.array(conn.get(node).data())
                # rho and time dims — ZIPFIT: dim0=rho, dim1=time
                rho = np.array(conn.get(f'DIM_OF({node}, 0)').data())
                t   = np.array(conn.get(f'DIM_OF({node}, 1)').data())
                conn.closeAllTrees()
                if t.size == 0 or d.size == 0:
                    continue
                if np.max(np.abs(t)) < 100:
                    t = t * 1e3   # s → ms
                # Ensure d is (n_rho, n_t)
                if d.ndim == 2:
                    if d.shape[0] != len(rho) and d.shape[1] == len(rho):
                        d = d.T
                d = d * scale
                results[key] = {'data': d.tolist(), 'rho': rho.tolist(), 'time': t.tolist()}
                print(f'  {key}: {node} ({d.shape})')
                break
            except Exception:
                continue

    # ── Ion profiles (IONS tree) ─────────────────────────────────────────────
    ion_candidates = {
        'Ti_keV': [
            ('\\IONS::TOP.PROFILE_FITS.ZIPFIT:ITEMPFIT',      'IONS', 1.0),   # keV
            ('\\IONS::TOP.PROFILE_FITS.ZIPFIT_CERA:ITEMPFIT', 'IONS', 1.0),   # keV
        ],
        'rot_kms': [
            ('\\IONS::TOP.PROFILE_FITS.ZIPFIT_CERA:TROTFIT', 'IONS', 1.0),
            ('\\IONS::TOP.PROFILE_FITS.ZIPFIT:TROTFIT',      'IONS', 1.0),
        ],
    }
    for key, candidates in ion_candidates.items():
        for node, tree, scale in candidates:
            try:
                conn = mds.Connection(ATLAS)
                conn.openTree(tree, shot)
                d = np.array(conn.get(node).data())
                # TROTFIT/ITEMPFIT: dim_of(sig,0)=rho, dim_of(sig,1)=time (per common.py)
                rho = np.array(conn.get(f'DIM_OF({node}, 0)').data())
                t   = np.array(conn.get(f'DIM_OF({node}, 1)').data())
                conn.closeAllTrees()
                if t.size == 0 or d.size == 0:
                    continue
                if np.max(np.abs(t)) < 100:
                    t = t * 1e3
                # numpy gives (n_rho, n_t) from MDSplus; normalise to (n_rho, n_t)
                if d.ndim == 2:
                    if d.shape[0] == len(t) and d.shape[1] == len(rho):
                        d = d.T  # (n_t, n_rho) → (n_rho, n_t)
                d = d * scale
                results[key] = {'data': d.tolist(), 'rho': rho.tolist(), 'time': t.tolist()}
                print(f'  {key}: {node} ({d.shape})')
                break
            except Exception:
                continue

    if not results:
        print(f'  Warning: no Thomson/CER profiles found for {shot}')
    return results


def _interp_profile_to_33(sig_dict, rho33, t_ms):
    """
    Interpolate a 2-D profile {data(n_rho, n_t), rho, time} to 33-pt rho33 at t_ms.
    Returns zero array if data unavailable.
    """
    if not sig_dict or not sig_dict.get('data'):
        return np.zeros(33)
    t   = np.asarray(sig_dict['time'])
    rho = np.asarray(sig_dict['rho'])
    d   = np.asarray(sig_dict['data'])
    if d.ndim == 1:
        return np.interp(rho33, rho, d)
    # 2-D: normalise to (n_rho, n_t)
    if d.shape[0] == len(t) and d.shape[1] == len(rho):
        d = d.T  # (n_t, n_rho) → (n_rho, n_t)
    i = np.searchsorted(t, t_ms)
    i = np.clip(i, 0, len(t) - 1)
    return np.interp(rho33, rho, d[:, i])


def run_tm_inference(shot, data_dir):
    """
    Run the DSM TM survival model (rt_model_latest_pcb.pkl) for one shot.

    Uses a subprocess call to the conda python3 environment which has
    torch + auton-survival installed. Loads all inputs from cached pkl files
    (no live MDSplus access needed).

    Input feature vector: 38 values (14 Z-scored scalars + 6 profiles × 4 PCA components).
    Preprocessing matches the rt_x_pca_bms_pcb.pkl training pipeline exactly.

    Parameters
    ----------
    shot     : int  shot number
    data_dir : str  directory containing {shot}/ subdirectory with cached pkl files

    Returns
    -------
    dict with 'time' (list of ms), 'data' (risk at 500 ms), 'horizons', 'risk'
    """
    import subprocess, tempfile

    for path, label in [
        (TM_MODEL_PKL,    'Model pkl'),
        (TM_NORM_PKL,     'Normalization pkl'),
        (TM_PCA_PKL,      'PCA pkl'),
        (DSM_INFER_SCRIPT,'Inference script'),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{label} not found: {path}')

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        out_pkl = tmp.name

    try:
        result = subprocess.run(
            [DSM_CONDA_PY, DSM_INFER_SCRIPT,
             str(shot), data_dir, TM_MODEL_PKL, TM_NORM_PKL, TM_PCA_PKL, out_pkl],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'DSM inference failed (rc={result.returncode}):\n'
                f'stdout: {result.stdout[-2000:]}\n'
                f'stderr: {result.stderr[-2000:]}'
            )
        print(result.stdout.strip())
        with open(out_pkl, 'rb') as fh:
            return pickle.load(fh)
    finally:
        if os.path.exists(out_pkl):
            os.unlink(out_pkl)


def run_tm_inference_cake(shot, data_dir):
    """
    Run DSM TM model_25 (42-feature CAKENN) for one shot.

    Uses dsm_infer_cakenn.py with model_25.pkl + cake_normalizations_dict.pkl +
    cakenn_pca_components.pkl. Evaluates on 5 ms grid (zero-order EFIT hold).
    For comparison against run_tm_inference() (rt_model_latest_pcb, valid_loss 0.1855).
    model_25 valid_loss = 0.5003.
    """
    import subprocess, tempfile

    for path, label in [
        (TM_CAKE_MODEL_PKL,       'CAKE model pkl'),
        (TM_CAKE_NORM_PKL,        'CAKE norm pkl'),
        (TM_CAKE_PCA_PKL,         'CAKE PCA pkl'),
        (DSM_INFER_CAKE_SCRIPT,   'CAKE inference script'),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{label} not found: {path}')

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        out_pkl = tmp.name

    try:
        result = subprocess.run(
            [DSM_CONDA_PY, DSM_INFER_CAKE_SCRIPT,
             str(shot), data_dir,
             TM_CAKE_MODEL_PKL, TM_CAKE_NORM_PKL, TM_CAKE_PCA_PKL, out_pkl],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'CAKE DSM inference failed (rc={result.returncode}):\n'
                f'stdout: {result.stdout[-2000:]}\n'
                f'stderr: {result.stderr[-2000:]}'
            )
        print(result.stdout.strip())
        with open(out_pkl, 'rb') as fh:
            return pickle.load(fh)
    finally:
        if os.path.exists(out_pkl):
            os.unlink(out_pkl)


def run_tm_inference_c(shot, data_dir):
    """
    Run TAPE C (k2c) rtProfiles model offline — same weights as PCS during the shot.

    Uses tape_c_infer.py → tape_offline binary (compiled from tape_rtprofile_*.c).
    Same 38-feature BMS-PCB preprocessing as run_tm_inference().
    Outputs survival (not risk); key 'data' = survival@500ms.

    NOTE: tape_rtprofile C weights are a separately trained model from
    rt_model_latest_pcb.pkl. This comparison shows PCS actual vs offline Python model.
    """
    import subprocess, tempfile

    for path, label in [
        (TAPE_C_INFER_SCRIPT, 'C infer script'),
        (TAPE_OFFLINE_BIN,    'tape_offline binary'),
        (TM_NORM_PKL,         'Normalization pkl'),
        (TM_PCA_PKL,          'PCA pkl'),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{label} not found: {path}')

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        out_pkl = tmp.name

    try:
        result = subprocess.run(
            [DSM_CONDA_PY, TAPE_C_INFER_SCRIPT,
             str(shot), data_dir, TM_NORM_PKL, TM_PCA_PKL, out_pkl],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'TAPE C inference failed (rc={result.returncode}):\n'
                f'stdout: {result.stdout[-2000:]}\n'
                f'stderr: {result.stderr[-2000:]}'
            )
        print(result.stdout.strip())
        with open(out_pkl, 'rb') as fh:
            return pickle.load(fh)
    finally:
        if os.path.exists(out_pkl):
            os.unlink(out_pkl)


def fetch_prediction(shot, signal, data_dir):
    try:
        d, t = fetch_ptdata(shot, signal)
        return {'data': d.tolist(), 'time': t.tolist()}
    except Exception as e:
        print(f'  Warning prediction signal {signal!r}: {e}')
        local = os.path.join(_shot_dir(shot, data_dir), f'{shot}_prediction.pkl')
        if os.path.exists(local):
            with open(local, 'rb') as fh:
                return pickle.load(fh)
        return {'data': [], 'time': []}

# ── pkl loading ────────────────────────────────────────────────────────────────

class _Stub:
    """Placeholder for unknown classes (e.g. xarray.DataArray) during unpickling."""
    def __init__(self, *a, **kw): pass
    def __setstate__(self, s):
        if isinstance(s, dict):
            self.__dict__.update(s)

class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub

def load_pkl_slice(shot, time_ms):
    """
    Load one time-slice pkl from the pre-computed MRE data directory.

    Parameters
    ----------
    shot    : int  shot number
    time_ms : int  time stamp in ms (must match a file name, e.g. 2500)

    Returns
    -------
    dict with keys:
      psiN, rhoN (129,)          — normalized flux / radius grids
      q_prof, p_prof (129,)      — safety factor, pressure [Pa]
      a_prof, Bp (129,)          — minor radius [m], poloidal field [T]
      R0, Bphi0, betaP           — scalars (m, T, dimensionless)
      eta_prof (201,)            — resistivity [Ω·m] on rho_201 grid
      J_BS (201,)                — bootstrap current density [A/m²]
      coll_i (201,192)           — ion collisionality vs (rho_201, t_full)
      rhostar (201,192)          — ion ρ* vs (rho_201, t_full)
      coll_i_dim_rho (201,)      — rho grid for 201-pt arrays
      coll_i_dim_t (192,)        — time axis [ms] for full-shot 2-D arrays
      eccd_prof (3,201)          — ECCD current density per gyrotron [A/m²?]
      peak_R_{0,1,2} (192,)      — ECCD deposition R [m] vs time per gyrotron
      peak_Z_{0,1,2} (192,)      — ECCD deposition Z [m] vs time per gyrotron
      peak_R_t_{0,1,2} (192,)   — time axis [ms] for deposition arrays
      rho_grid (129,129)         — rho on (R,Z) MHD grid
      R_grid, Z_grid (129,)      — MHD grid [m]
      ne, Te                     — xarray stubs (require xarray to fully load)
    """
    p = os.path.join(PKL_DIR, str(shot), f'{int(time_ms)}.pkl')
    with open(p, 'rb') as f:
        return _PermissiveUnpickler(f).load()

def pkl_times(shot):
    """Return sorted list of available time stamps (int ms) for a shot."""
    d = os.path.join(PKL_DIR, str(shot))
    times = []
    for f in os.listdir(d):
        if f.endswith('.pkl'):
            stem = f[:-4]
            if stem.isdigit():   # skip cache files like {shot}_{key}.pkl
                times.append(int(stem))
    return sorted(times)

def load_all_pkl(shot, verbose=False):
    """
    Load all pkl time-slices for one shot.

    Returns
    -------
    dict  {time_ms (int): data_dict}
    """
    times = pkl_times(shot)
    slices = {}
    for t in times:
        slices[t] = load_pkl_slice(shot, t)
        if verbose:
            print(f'  loaded t={t} ms')
    return slices

def pkl_scalar_series(slices, key):
    """Extract a scalar field (R0, betaP, …) across all time slices."""
    times = sorted(slices)
    vals  = [float(slices[t][key]) for t in times]
    return np.array(times, dtype=float), np.array(vals)

def pkl_profile_at_rho(slices, key, rho_target, rho_key='rhoN', grid_201=False):
    """
    Interpolate a 1-D profile field to rho_target at each time slice.

    Parameters
    ----------
    key        : field name (e.g. 'q_prof', 'eta_prof', 'J_BS')
    rho_target : float  normalized rho at which to evaluate
    rho_key    : 'rhoN' for 129-pt arrays; 'coll_i_dim_rho' for 201-pt arrays
    grid_201   : if True use coll_i_dim_rho (201 pts) instead of rhoN

    Returns
    -------
    times_ms (N,), values (N,)
    """
    times = sorted(slices)
    vals  = []
    for t in times:
        d = slices[t]
        v = np.asarray(d[key])
        rho = np.asarray(d['coll_i_dim_rho' if grid_201 else rho_key])
        vals.append(float(np.interp(rho_target, rho, v)))
    return np.array(times, dtype=float), np.array(vals)

# ── MRE equilibrium helpers ────────────────────────────────────────────────────

def find_q_surface_pkl(d, q_target=2.0, rho_min=0.25):
    """
    Find minor radius r_s where q = q_target from a pkl slice.

    Returns (rho_s, r_s) or (None, None) if q_target not crossed.
    rho_s is normalized (0–1), r_s is in metres.

    Takes the outermost crossing with rho_s > rho_min to skip reversed-shear
    inner crossings and axis-grazing spikes.
    """
    rhoN   = np.asarray(d['rhoN'])
    q_prof = np.asarray(d['q_prof'])
    a_prof = np.asarray(d['a_prof'])
    crossings = np.where(np.diff(np.sign(q_prof - q_target)))[0]
    if len(crossings) == 0:
        return None, None
    # resolve each crossing to rho and keep only those outside rho_min
    valid_i = []
    for idx in crossings:
        frac  = (q_target - q_prof[idx]) / (q_prof[idx+1] - q_prof[idx])
        rho_c = rhoN[idx] + frac * (rhoN[idx+1] - rhoN[idx])
        if rho_c >= rho_min:
            valid_i.append(idx)
    if not valid_i:
        return None, None
    i = valid_i[-1]   # outermost qualifying crossing
    frac  = (q_target - q_prof[i]) / (q_prof[i+1] - q_prof[i])
    rho_s = rhoN[i] + frac * (rhoN[i+1] - rhoN[i])
    r_s   = float(np.interp(rho_s, rhoN, a_prof))
    return float(rho_s), r_s

def mre_quantities_from_pkl(d, q_target=2.0):
    """
    Compute MRE-relevant scalar quantities from one pkl time slice.

    Returns dict with:
      rho_s, r_s        — q=q_target surface location
      Bp_s              — poloidal field at r_s [T]
      eta_s             — resistivity at r_s [Ω·m]
      tau_R             — resistive time μ₀r_s²/(1.22η) [s]
      J_BS_s            — bootstrap current density at r_s [A/m²]
      betaP             — scalar poloidal beta
      Lp, Lq            — pressure and q scale lengths [m]  (None if not computable)
      eccd_at_s (3,)    — ECCD current density per gyrotron at r_s [A/m²]
    Returns None if q=2 surface not found.
    """
    MU0 = 4 * np.pi * 1e-7

    rhoN   = np.asarray(d['rhoN'])
    q_prof = np.asarray(d['q_prof'])
    p_prof = np.asarray(d['p_prof'])
    a_prof = np.asarray(d['a_prof'])
    Bp     = np.asarray(d['Bp'])

    # a_prof is only populated for the inner fraction of the grid; extrapolate
    # linearly through the valid (non-zero) points to get r(rho) everywhere.
    valid = a_prof > 1e-6
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(rhoN[valid], a_prof[valid], 1)
        r_phys = np.clip(rhoN * slope + intercept, 0.0, None)
    else:
        r_phys = a_prof.copy()

    rho_s, _ = find_q_surface_pkl(d, q_target)
    if rho_s is None:
        return None
    r_s = float(np.interp(rho_s, rhoN, r_phys))

    rho201 = np.asarray(d['coll_i_dim_rho'])
    eta    = np.asarray(d['eta_prof'])
    J_BS   = np.asarray(d['J_BS'])

    Bp_s   = float(np.interp(rho_s, rhoN, Bp))
    eta_s  = float(np.interp(rho_s, rho201, eta))
    tau_R  = MU0 * r_s**2 / (1.22 * eta_s) if eta_s > 0 else None
    J_BS_s = float(np.interp(rho_s, rho201, J_BS))

    # Scale lengths at r_s.  Lp = -p/(dp/dr) > 0 (p decreasing outward).
    # Lq = +q/(dq/dr) > 0 (q increasing outward) — note POSITIVE sign.
    def _L(prof, sign=-1):
        ddr = np.gradient(prof, r_phys)
        f_s  = float(np.interp(rho_s, rhoN, prof))
        df_s = float(np.interp(rho_s, rhoN, ddr))
        if df_s == 0 or f_s == 0:
            return None
        L = abs(sign * f_s / df_s)
        return L if L > 1e-10 else None
    Lp = _L(p_prof, sign=-1)   # pressure decreases → df_s < 0 → -f/df > 0
    Lq = _L(q_prof, sign=+1)   # q increases → df_s > 0 → +f/df > 0

    # ECCD current density at q=2 surface (3 gyrotrons × 201-pt rho grid)
    eccd = np.asarray(d['eccd_prof'])   # (3, 201)
    eccd_at_s = np.array([float(np.interp(rho_s, rho201, eccd[i])) for i in range(3)])

    return {
        'rho_s':    rho_s,
        'r_s':      r_s,
        'Bp_s':     Bp_s,
        'eta_s':    eta_s,
        'tau_R':    tau_R,
        'J_BS_s':   J_BS_s,
        'betaP':    float(d['betaP']),
        'Lp':       Lp,
        'Lq':       Lq,
        'eccd_at_s': eccd_at_s,
    }

def mre_timeseries(slices, q_target=2.0):
    """
    Compute mre_quantities_from_pkl for every time slice.

    Returns
    -------
    times_ms : ndarray (N,)
    mre_ts   : list of dicts (or None where q surface not found)
    """
    times = sorted(slices)
    results = []
    for t in times:
        results.append(mre_quantities_from_pkl(slices[t], q_target))
    return np.array(times, dtype=float), results

# ── Analysis utilities ─────────────────────────────────────────────────────────

def check_tm_events(data, time, thresh_G, min_dur_ms, smooth_ms=100.0):
    """Find segments where the rolling-maximum envelope of |data| exceeds thresh_G
    for >= min_dur_ms.  smooth_ms sets the rolling-max window to handle the
    amplitude modulation of a rotating mode (N1RMS at 5 ms resolution oscillates
    at the rotation frequency, so raw contiguous threshold checks fail)."""
    data, time = np.asarray(data, float), np.asarray(time, float)
    amp = np.abs(data)

    # Causal rolling maximum over smooth_ms look-back window.
    # Envelope[i] = max(amp[i-w:i+1]) so the envelope rises as soon as the
    # signal crosses threshold, avoiding a non-causal shift of the onset time.
    dt_ms = float(np.median(np.diff(time))) if len(time) > 1 else 5.0
    w = max(1, int(round(smooth_ms / dt_ms)))
    n = len(amp)
    envelope = np.array([amp[max(0, i - w):i + 1].max() for i in range(n)])

    above = envelope > thresh_G
    events, in_event, t_start, i_start = [], False, None, None
    for i in range(len(time)):
        if above[i] and not in_event:
            in_event, t_start, i_start = True, time[i], i
        elif not above[i] and in_event:
            dur = time[i-1] - t_start
            if dur >= min_dur_ms:
                events.append({'t_start': float(t_start), 't_end': float(time[i-1]),
                               'duration_ms': float(dur),
                               'peak_G': float(amp[i_start:i].max())})
            in_event = False
    if in_event:
        dur = time[-1] - t_start
        if dur >= min_dur_ms:
            events.append({'t_start': float(t_start), 't_end': float(time[-1]),
                           'duration_ms': float(dur),
                           'peak_G': float(amp[i_start:].max())})
    return events

def get_tm_onset(n1rms_dict, t_win, thresh_G, min_dur_ms):
    data = np.array(n1rms_dict.get('data', []))
    time = np.array(n1rms_dict.get('time', []))
    if data.size == 0:
        return None
    mask = (time >= t_win[0]) & (time <= t_win[1])
    evts = check_tm_events(data[mask], time[mask], thresh_G, min_dur_ms)
    return evts[0]['t_start'] if evts else None

def band_rms(data, time_ms, f_lo=5e3, f_hi=25e3):
    """Band-pass filter (f_lo–f_hi Hz) and return instantaneous amplitude."""
    dt = np.median(np.diff(time_ms)) * 1e-3   # ms → s
    fs = 1.0 / dt
    nyq = 0.5 * fs
    lo, hi = f_lo / nyq, min(f_hi / nyq, 0.99)
    if lo >= hi or lo <= 0:
        return np.abs(data)
    b, a = scipy_signal.butter(4, [lo, hi], btype='band')
    return np.abs(scipy_signal.filtfilt(b, a, data))

def compute_br_tilde(mpi_dict, t_win):
    """
    RMS envelope of band-filtered MPI signals as a proxy for |B̃_r| [Gauss].
    Returns (time_ms, amplitude_G).
    """
    all_env, t_ref = [], None
    for arr in mpi_dict.values():
        t = np.array(arr.get('time', []))
        v = np.array(arr.get('data', []))
        if t.size == 0:
            continue
        mask = (t >= t_win[0]) & (t <= t_win[1])
        if mask.sum() < 10:
            continue
        all_env.append(band_rms(v[mask], t[mask]))
        if t_ref is None:
            t_ref = t[mask]
    if not all_env or t_ref is None:
        return np.array([]), np.array([])
    n = min(len(r) for r in all_env)
    return t_ref[:n], np.sqrt(np.mean(np.vstack([r[:n] for r in all_env])**2, axis=0))
