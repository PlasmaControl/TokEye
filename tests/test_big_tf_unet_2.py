"""Unit tests for the big_tf_unet_2 workflow layer + math core.

Collection-safe under `uv sync --dev` (no train deps): modules that pull
h5py/kneed are guarded with importorskip.
"""

from __future__ import annotations

import pytest

from tokeye.training.big_tf_unet_2.paths import (
    STEP_ORDER,
    RunPaths,
    get_step,
    run_id_for,
    steps_after,
)
from tokeye.training.big_tf_unet_2.run_config import (
    ConfigError,
    check_scale_lock,
    load_run_config,
)
from tokeye.training.big_tf_unet_2.task_matrix import RunTaskMatrix, params_hash

MODS = ["ece", "mhr", "bes"]


def _write_run_yaml(path, body="run: {nfft: 512, hop: 128}\n"):
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_run_id_and_registry():
    assert run_id_for(512, 128) == "nfft512_hop128"
    assert run_id_for(512, 128, "v2") == "nfft512_hop128_v2"
    assert STEP_ORDER[0] == "step_0" and STEP_ORDER[-1] == "step_8"
    assert get_step("step_3").exec_mode == "sbatch_gpu"
    assert not get_step("step_5").per_modality
    assert [s.name for s in steps_after("step_6")] == ["step_7", "step_8"]
    with pytest.raises(KeyError):
        get_step("step_99")


def test_artifacts_are_registered_per_step(tmp_path):
    paths = RunPaths("t", root=tmp_path)
    per_mod = paths.artifacts("step_2", ["ece"])
    assert paths.step_h5("step_2", "ece") in per_mod
    assert paths.baseline_h5("ece") in per_mod
    combined = paths.artifacts("step_5", MODS)
    assert combined == [paths.step_h5("step_5")]
    assert paths.artifacts("step_7", MODS) == [paths.model_dir]


# ---------------------------------------------------------------------------
# run_config
# ---------------------------------------------------------------------------


def test_valid_config_loads(tmp_path):
    cfg = load_run_config(_write_run_yaml(tmp_path / "run.yaml"))
    assert cfg.run.nfft == 512
    assert cfg.n_freq == 257
    assert cfg.modality_names == MODS


@pytest.mark.parametrize(
    "body",
    [
        "run: {nfft: 512, hop: 1024}",  # hop > nfft
        "run: {nfft: 500, hop: 100}",  # off-grid
        "run: {nfft: 512, hop: 128}\nrefine: {model_trust: 2.0}",  # out of range
        "run: {nfft: 512, hop: 128}\nlabels: {knee_sensitivty: 1.0}",  # typo
    ],
)
def test_bad_configs_raise_config_error(tmp_path, body):
    with pytest.raises(ConfigError):
        load_run_config(_write_run_yaml(tmp_path / "run.yaml", body))


def test_custom_scale_needs_opt_in(tmp_path):
    body = "run: {nfft: 500, hop: 100, allow_custom_scale: true}"
    cfg = load_run_config(_write_run_yaml(tmp_path / "run.yaml", body))
    assert cfg.run.nfft == 500


def test_scale_lock(tmp_path):
    cfg = load_run_config(_write_run_yaml(tmp_path / "run.yaml"))
    meta = tmp_path / "run_meta.json"
    meta.write_text('{"nfft": 1024, "hop": 256}')
    with pytest.raises(ConfigError):
        check_scale_lock(cfg, meta)
    meta.write_text('{"nfft": 512, "hop": 128}')
    check_scale_lock(cfg, meta)  # no raise


# ---------------------------------------------------------------------------
# task matrix
# ---------------------------------------------------------------------------


def test_staleness_propagates_per_modality(tmp_path):
    m = RunTaskMatrix(tmp_path / "tm.json")
    h = params_hash({"x": 1})
    m.mark_complete("step_0", "ece", h, MODS)
    m.mark_complete("step_1", "ece", h, MODS)
    m.mark_complete("step_1", "mhr", h, MODS)
    m.mark_complete("step_5", None, h, MODS)
    # rerun of ece step_0 stales ece's chain + combined steps, NOT mhr's
    m.mark_complete("step_0", "ece", params_hash({"x": 2}), MODS)
    assert m.status("step_1", "ece") == "stale"
    assert m.status("step_1", "mhr") == "complete"
    assert m.status("step_5") == "stale"


def test_accept_requires_complete(tmp_path):
    m = RunTaskMatrix(tmp_path / "tm.json")
    h = params_hash({})
    m.mark_complete("step_0", "ece", h, MODS)
    with pytest.raises(ValueError, match="not complete"):
        m.accept("step_0", MODS)
    for mod in MODS[1:]:
        m.mark_complete("step_0", mod, h, MODS)
    m.accept("step_0", MODS)
    assert m.is_accepted("step_0", MODS)
    # rerunning revokes acceptance
    m.mark_complete("step_0", "ece", h, MODS)
    assert not m.is_accepted("step_0", MODS)


def test_clear_resets_and_stales(tmp_path):
    m = RunTaskMatrix(tmp_path / "tm.json")
    h = params_hash({})
    for mod in MODS:
        m.mark_complete("step_0", mod, h, MODS)
        m.mark_complete("step_1", mod, h, MODS)
    m.mark_pending("step_0", MODS)
    assert m.status("step_0", "ece") == "pending"
    assert m.status("step_1", "ece") == "stale"


# ---------------------------------------------------------------------------
# clearing (fence)
# ---------------------------------------------------------------------------


def test_clear_fence_refuses_outside_roots(tmp_path):
    from tokeye.training.big_tf_unet_2.clearing import _assert_fenced

    paths = RunPaths("t", root=tmp_path)
    inside = paths.cache_root / "ece" / "step_0.h5"
    _assert_fenced(inside, [paths.cache_root])  # no raise
    with pytest.raises(RuntimeError, match="Refusing"):
        _assert_fenced(tmp_path / "outside.txt", [paths.cache_root])


# ---------------------------------------------------------------------------
# math core (train deps required)
# ---------------------------------------------------------------------------


def test_normalize_asinh_properties():
    ap = pytest.importorskip("tokeye.training.big_tf_unet_2.utils.auto_params")
    np = pytest.importorskip("numpy")
    x = np.random.default_rng(0).normal(0, 1, 50_000)
    med, scale = ap.robust_stats(x)
    n3 = ap.normalize_asinh(x, 3.0, med, scale)
    bulk = np.abs(x) < 0.5
    assert np.allclose(n3[bulk], (x[bulk] - med) / scale, atol=0.02)
    grid = ap.normalize_asinh(np.linspace(-100, 100, 1000), 3.0, 0.0, 1.0)
    assert np.all(np.diff(grid) > 0)  # strictly monotone (invertible)


def test_knee_threshold_synthetic():
    pytest.importorskip("kneed")
    ap = pytest.importorskip("tokeye.training.big_tf_unet_2.utils.auto_params")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 200_000)
    z[:2000] = rng.normal(8, 0.5, 2000)  # 1% strong signal
    r = ap.knee_threshold(z)
    assert not r["used_fallback"]
    assert 1.0 < r["threshold"] < 8.0
    r2 = ap.knee_threshold(z, delta=1.5)
    assert r2["threshold"] == pytest.approx(r["threshold"] + 1.5)
    degenerate = ap.knee_threshold(np.full(10, -1.0))
    assert degenerate["used_fallback"]


def test_edge_bins_energy_catches_plateau(tmp_path):
    h5py = pytest.importorskip("h5py")
    ap = pytest.importorskip("tokeye.training.big_tf_unet_2.utils.auto_params")
    np = pytest.importorskip("numpy")
    n_freq = 257
    profile = np.ones(n_freq)
    profile[:35] = 40.0  # broad low-frequency plateau (the bes failure mode)
    profile[-3:] = 30.0
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("samples")
        for i in range(4):
            noise = np.random.default_rng(i).normal(1, 0.05, (2, n_freq, 64, 2))
            grp.create_dataset(str(i), data=(profile[None, :, None, None] * noise))
    lower, upper = ap.detect_edge_bins_energy(path, k=2.0)
    assert 33 <= lower <= 38  # full plateau; gradient method finds ~1
    assert 2 <= upper <= 5


def test_scale_covariant_autos():
    ap = pytest.importorskip("tokeye.training.big_tf_unet_2.utils.auto_params")
    assert ap.compute_lam(513) == pytest.approx(1.0e6)
    assert 0.05e6 < ap.compute_lam(257) < 0.07e6  # (257/513)^4
    assert ap.compute_num_layers(257, 516) == 5
    assert ap.compute_num_layers(65, 1032) == 4  # nfft=128 shrinks the net
    assert ap.compute_batch_size(8, 257, 516, 5) >= 8


# ---------------------------------------------------------------------------
# scaffold (isolated cwd)
# ---------------------------------------------------------------------------


def test_scaffold_creates_and_refuses_overwrite(tmp_path, monkeypatch):
    from tokeye.training.big_tf_unet_2.scaffold import scaffold_run

    # repo_root() walks up for pyproject.toml; give the sandbox one so the
    # scaffold lands inside tmp_path instead of the real checkout.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sandbox'\n")
    monkeypatch.chdir(tmp_path)
    paths = scaffold_run(512, 128)
    assert paths.root == tmp_path
    assert paths.run_yaml.exists()
    assert paths.run_meta.exists()
    cfg = load_run_config(paths.run_yaml)
    check_scale_lock(cfg, paths.run_meta)
    with pytest.raises(FileExistsError):
        scaffold_run(512, 128)
