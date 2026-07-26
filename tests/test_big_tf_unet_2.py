"""Unit tests for the big_tf_unet_2 workflow layer + math core.

Collection-safe under `uv sync --dev` (no train deps): modules that pull
h5py/kneed are guarded with importorskip.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tokeye.training.big_tf_unet_2.paths import (
    STEP_ORDER,
    STEPS,
    RunPaths,
    get_step,
    run_id_for,
    step_for_config,
    steps_after,
    template_dir,
)
from tokeye.training.big_tf_unet_2.run_config import (
    ConfigError,
    check_scale_lock,
    config_sources,
    load_run_config,
)
from tokeye.training.big_tf_unet_2.task_matrix import RunTaskMatrix, params_hash

MODS = ["ece", "mhr", "bes", "co2"]
TEMPLATE = template_dir(Path(__file__).resolve().parents[1])


def _runner():
    """The runner imports auto_resolve -> auto_params -> h5py, a train-group dep.
    CI installs `--dev --extra app` only, so skip rather than error there."""
    return pytest.importorskip("tokeye.training.big_tf_unet_2.runner")


def _workspace(tmp_path, **step_bodies):
    """A minimal run folder: run.yml plus any step ymls asked for."""
    ws = tmp_path / "dev" / "training" / "nfft512_hop128"
    ws.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sandbox'\n")
    (ws / "run.yml").write_text("run: {nfft: 512, hop: 128}\n")
    for stem, body in step_bodies.items():
        (ws / f"{stem}.yml").write_text(body)
    return ws


# ---------------------------------------------------------------------------
# paths + step registry
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


def test_every_step_owns_a_distinct_knob_section():
    """The triplet naming depends on one section per step; guard the invariant."""
    sections = [s.knob_section for s in STEPS]
    assert len(set(sections)) == len(STEPS)


def test_artifacts_are_registered_per_step(tmp_path):
    paths = RunPaths("t", root=tmp_path)
    per_mod = paths.artifacts("step_2", ["ece"])
    assert paths.step_h5("step_2", "ece") in per_mod
    assert paths.baseline_h5("ece") in per_mod
    combined = paths.artifacts("step_5", MODS)
    assert combined == [paths.step_h5("step_5")]
    assert paths.artifacts("step_7", MODS) == [paths.model_dir]


def test_log_dir_is_the_flat_repo_logs(tmp_path):
    paths = RunPaths("t", root=tmp_path)
    assert paths.log_dir == tmp_path / "logs"
    assert not paths.log_dir.is_relative_to(paths.cache_root)


def test_workspace_filenames_come_from_the_registry(tmp_path):
    paths = RunPaths("t", root=tmp_path)
    assert paths.run_yml.name == "run.yml"
    assert paths.step_yml("step_3").name == "step_3_denoise.yml"
    assert paths.step_sh("step_3").name == "step_3_denoise.sh"
    assert paths.step_ipynb("step_3").name == "step_3_denoise.ipynb"


@pytest.mark.parametrize("spec", STEPS, ids=lambda s: s.name)
def test_step_inferred_from_config_stem(spec):
    assert step_for_config(f"/anywhere/{spec.module}.yml").name == spec.name


def test_step_inference_rejects_a_non_step_config():
    with pytest.raises(KeyError, match="not a step config"):
        step_for_config("/anywhere/run.yml")


# ---------------------------------------------------------------------------
# the run-folder template
# ---------------------------------------------------------------------------


def test_template_has_a_triplet_per_step():
    assert (TEMPLATE / "run.yml").is_file()
    for spec in STEPS:
        for ext in ("yml", "sh", "ipynb"):
            assert (TEMPLATE / f"{spec.module}.{ext}").is_file(), spec.module
    # Flat: no subdirectories in a run folder.
    assert not [p for p in TEMPLATE.iterdir() if p.is_dir()]


@pytest.mark.parametrize("spec", STEPS, ids=lambda s: s.name)
def test_template_yml_owns_its_own_section(spec):
    body = (TEMPLATE / f"{spec.module}.yml").read_text()
    sections = re.findall(r"^([a-z_]+):", body, flags=re.MULTILINE)
    assert sections == [spec.knob_section], sections


@pytest.mark.parametrize("spec", STEPS, ids=lambda s: s.name)
def test_template_sh_is_consistent_with_the_registry(spec):
    body = (TEMPLATE / f"{spec.module}.sh").read_text()
    # Names its own yml, and only its own.
    assert f'"$SLURM_SUBMIT_DIR/{spec.module}.yml"' in body
    assert len(re.findall(r"--config", body)) == 1
    # --chdir is what makes logs/ and .venv/ resolve from a static template.
    assert "#SBATCH --chdir=" in body
    assert "#SBATCH --output=logs/" in body
    # GPU iff the registry says so; CPU work must never hold a GPU.
    assert ("--gres=gpu:1" in body) == (spec.exec_mode == "sbatch_gpu")
    # An array iff the step is per-modality, sized to run.yml's modality list.
    if spec.per_modality:
        assert f"#SBATCH --array=0-{len(MODS) - 1}" in body
        assert f"MODS=({' '.join(MODS)})" in body
    else:
        assert "--array" not in body
    # No srun: single-task jobs, and it keeps the parent job environment out.
    assert "srun" not in body


def test_template_run_yml_modalities_match_the_array_ranges():
    cfg = load_run_config(TEMPLATE / "run.yml")
    assert cfg.modality_names == MODS


# ---------------------------------------------------------------------------
# run_config merge chain
# ---------------------------------------------------------------------------


def test_config_sources_pairs_a_step_with_run_yml(tmp_path):
    ws = _workspace(tmp_path, step_2_baseline="baseline: {edge_k: 3.0}\n")
    assert config_sources(ws / "run.yml") == [ws / "run.yml"]
    assert config_sources(ws / "step_2_baseline.yml") == [
        ws / "run.yml",
        ws / "step_2_baseline.yml",
    ]


def test_step_yml_merges_over_run_yml_over_defaults(tmp_path):
    ws = _workspace(tmp_path, step_2_baseline="baseline: {edge_k: 3.0}\n")
    cfg = load_run_config(ws / "step_2_baseline.yml")
    assert cfg.baseline.edge_k == 3.0  # from the step file
    assert cfg.run.nfft == 512  # from run.yml
    assert cfg.modality_names == MODS  # from defaults
    assert cfg.baseline.method == "fabc"  # untouched default


def test_step_yml_can_override_a_global(tmp_path):
    """The merge chain is what makes a per-step `overwrite: false` free."""
    ws = _workspace(tmp_path, step_6_refine="overwrite: false\nrefine: {n_folds: 3}\n")
    assert load_run_config(ws / "run.yml").overwrite is True
    step_cfg = load_run_config(ws / "step_6_refine.yml")
    assert step_cfg.overwrite is False
    assert step_cfg.refine.n_folds == 3


def test_split_files_match_one_combined_file(tmp_path):
    ws = _workspace(tmp_path, step_4_labels="labels: {delta: 0.5}\n")
    split = load_run_config(ws / "step_4_labels.yml")
    combined = tmp_path / "combined.yml"
    combined.write_text("run: {nfft: 512, hop: 128}\nlabels: {delta: 0.5}\n")
    assert load_run_config(combined) == split


def test_valid_config_loads(tmp_path):
    ws = _workspace(tmp_path)
    cfg = load_run_config(ws / "run.yml")
    assert cfg.run.nfft == 512
    assert cfg.n_freq == 257
    assert cfg.modality_names == MODS
    assert cfg.overwrite is True


@pytest.mark.parametrize(
    "body",
    [
        "run: {nfft: 512, hop: 1024}",  # hop > nfft
        "run: {nfft: 500, hop: 100}",  # off-grid
        "run: {nfft: 512, hop: 128}\nrefine: {model_trust: 2.0}",  # out of range
        "run: {nfft: 512, hop: 128}\nlabels: {knee_sensitivty: 1.0}",  # typo
        "run: {nfft: 512, hop: 128}\nslurm: {gpu_time: '5:00:00'}",  # section gone
    ],
)
def test_bad_configs_raise_config_error(tmp_path, body):
    path = tmp_path / "run.yml"
    path.write_text(body)
    with pytest.raises(ConfigError):
        load_run_config(path)


def test_missing_config_names_the_file(tmp_path):
    with pytest.raises(ConfigError, match="config not found"):
        load_run_config(tmp_path / "nope.yml")


def test_custom_scale_needs_opt_in(tmp_path):
    path = tmp_path / "run.yml"
    path.write_text("run: {nfft: 500, hop: 100, allow_custom_scale: true}")
    assert load_run_config(path).run.nfft == 500


def test_scale_lock(tmp_path):
    path = tmp_path / "run.yml"
    path.write_text("run: {nfft: 512, hop: 128}\n")
    cfg = load_run_config(path)
    meta = tmp_path / "run_meta.json"
    check_scale_lock(cfg, meta)  # absent lock: no-op until the first run
    meta.write_text('{"nfft": 1024, "hop": 256}')
    with pytest.raises(ConfigError):
        check_scale_lock(cfg, meta)
    meta.write_text('{"nfft": 512, "hop": 128}')
    check_scale_lock(cfg, meta)  # no raise


# ---------------------------------------------------------------------------
# runner: scale lock, overwrite, modality validation
# ---------------------------------------------------------------------------


def test_first_run_writes_the_scale_lock(tmp_path, monkeypatch):
    _lock_scale = _runner()._lock_scale

    ws = _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    paths = RunPaths(ws.name, root=tmp_path)
    cfg = load_run_config(ws / "run.yml")
    assert not paths.run_meta.exists()
    _lock_scale(paths, cfg)
    assert paths.run_meta.is_file()
    check_scale_lock(cfg, paths.run_meta)  # agrees with itself
    _lock_scale(paths, cfg)  # idempotent


def test_overwrite_false_refuses_when_artifacts_exist(tmp_path):
    _replace_artifacts = _runner()._replace_artifacts

    ws = _workspace(tmp_path, step_5_dataset="overwrite: false\n")
    paths = RunPaths(ws.name, root=tmp_path)
    cfg = load_run_config(ws / "step_5_dataset.yml")
    target = paths.step_h5("step_5")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("expensive")
    matrix = RunTaskMatrix(paths.task_matrix_path)
    with pytest.raises(ConfigError, match="overwrite is false"):
        _replace_artifacts(paths, cfg, "step_5", None, matrix, params_hash({"a": 1}))
    assert target.exists()  # refused, not clobbered


def test_overwrite_true_clears_only_its_own_modality(tmp_path):
    """The regression clearing.py exists to prevent: wiping sibling modalities."""
    _replace_artifacts = _runner()._replace_artifacts

    ws = _workspace(tmp_path)
    paths = RunPaths(ws.name, root=tmp_path)
    cfg = load_run_config(ws / "run.yml")
    matrix = RunTaskMatrix(paths.task_matrix_path)
    made = {}
    for mod in ("ece", "mhr"):
        target = paths.step_h5("step_2", mod)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(mod)
        made[mod] = target
    _replace_artifacts(paths, cfg, "step_2", "ece", matrix, params_hash({"a": 1}))
    assert not made["ece"].exists()
    assert made["mhr"].read_text() == "mhr"


# ---------------------------------------------------------------------------
# fold-level resume (step_6)
# ---------------------------------------------------------------------------


def test_step_6_is_the_only_resumable_step():
    assert get_step("step_6").resumable
    assert [s.name for s in STEPS if s.resumable] == ["step_6"]


def _partial_step_6(tmp_path):
    ws = _workspace(tmp_path)
    paths = RunPaths(ws.name, root=tmp_path)
    cfg = load_run_config(ws / "run.yml")
    partial = paths.step_h5("step_6")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("folds 0-2 of 5")
    return paths, cfg, RunTaskMatrix(paths.task_matrix_path), partial


def test_resumable_step_keeps_its_partial_when_knobs_unchanged(tmp_path):
    _replace_artifacts = _runner()._replace_artifacts

    paths, cfg, matrix, partial = _partial_step_6(tmp_path)
    same = params_hash({"max_epochs": 200})
    matrix.mark_running("step_6", None, job_id="1", step_params_hash=same)
    _replace_artifacts(paths, cfg, "step_6", None, matrix, same)
    assert partial.read_text() == "folds 0-2 of 5"


def test_resumable_step_restarts_when_a_knob_changed(tmp_path):
    """Half a result computed under different settings is not a result."""
    _replace_artifacts = _runner()._replace_artifacts

    paths, cfg, matrix, partial = _partial_step_6(tmp_path)
    matrix.mark_running(
        "step_6", None, job_id="1", step_params_hash=params_hash({"max_epochs": 200})
    )
    _replace_artifacts(
        paths, cfg, "step_6", None, matrix, params_hash({"max_epochs": 50})
    )
    assert not partial.exists()


def test_non_resumable_step_always_restarts(tmp_path):
    _replace_artifacts = _runner()._replace_artifacts

    ws = _workspace(tmp_path)
    paths = RunPaths(ws.name, root=tmp_path)
    cfg = load_run_config(ws / "run.yml")
    matrix = RunTaskMatrix(paths.task_matrix_path)
    same = params_hash({"a": 1})
    matrix.mark_running("step_5", None, job_id="1", step_params_hash=same)
    target = paths.step_h5("step_5")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("partial")
    _replace_artifacts(paths, cfg, "step_5", None, matrix, same)
    assert not target.exists()


def test_open_output_resumes_a_compatible_partial(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    refine = pytest.importorskip("tokeye.training.big_tf_unet_2.step_6_refine")

    out_path = tmp_path / "step_6.h5"
    shape = (2, 8, 8)

    out, done, metrics = refine._open_output(out_path, 10, shape, 5)
    assert (done, metrics) == ([], [])
    out["p_oof"][3] = np.ones(shape, dtype=np.float32)
    out.attrs["folds_done"] = "[0, 1]"
    out.attrs["fold_metrics"] = '[{"fold": 0}, {"fold": 1}]'
    out.close()

    out, done, metrics = refine._open_output(out_path, 10, shape, 5)
    assert done == [0, 1]
    assert [m["fold"] for m in metrics] == [0, 1]
    assert out["p_oof"][3].sum() == float(np.prod(shape))  # earlier work intact
    out.close()

    # A different fold count or sample count is not resumable — start clean.
    out, done, metrics = refine._open_output(out_path, 10, shape, 3)
    assert (done, metrics) == ([], [])
    assert out["p_oof"][3].sum() == 0.0
    out.close()


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


def test_no_sign_off_flag_remains(tmp_path):
    m = RunTaskMatrix(tmp_path / "tm.json")
    assert not hasattr(m, "accept")
    assert not hasattr(m, "is_accepted")
    m.mark_complete("step_0", "ece", params_hash({}), MODS)
    assert "accepted" not in m.to_rows(MODS)[0]


def test_running_records_the_slurm_job_id(tmp_path):
    """view.log() finds a step's output by the job id the job recorded itself."""
    m = RunTaskMatrix(tmp_path / "tm.json")
    m.mark_running("step_6", None, job_id="2884567")
    assert m.job_id("step_6") == "2884567"
    m.mark_running("step_6", None)  # a later call keeps the known id
    assert m.job_id("step_6") == "2884567"


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
# view (read-only surface)
# ---------------------------------------------------------------------------


def test_view_here_uses_the_cwd_as_the_run_id(tmp_path, monkeypatch):
    from tokeye.training.big_tf_unet_2 import view

    ws = _workspace(tmp_path)
    monkeypatch.chdir(ws)
    v = view.here()
    assert v.run_id == "nfft512_hop128"
    assert v.paths.root == tmp_path


def test_view_here_outside_a_run_folder_says_so(tmp_path, monkeypatch):
    from tokeye.training.big_tf_unet_2 import view

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="No run.yml"):
        view.here()


def test_view_cannot_submit_or_delete():
    from tokeye.training.big_tf_unet_2 import view

    for forbidden in ("submit", "run", "wait", "accept", "clear", "clear_all"):
        assert not hasattr(view.RunView, forbidden), forbidden


def test_view_has_no_module_level_train_only_imports():
    """CI installs `--dev --extra app` only, so `view` must import without
    pandas/h5py/matplotlib. gallery and auto_resolve pull those in, so they are
    deferred into the methods that use them."""
    import ast

    source = Path(view_module_path()).read_text()
    top_level = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            top_level |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level.add(node.module.split(".")[0])
            top_level |= {a.name for a in node.names}
    assert not top_level & {"pandas", "h5py", "matplotlib", "gallery", "auto_resolve"}


def view_module_path() -> str:
    from tokeye.training.big_tf_unet_2 import view

    return view.__file__


def test_view_finds_a_steps_logs_by_job_name(tmp_path, monkeypatch):
    from tokeye.training.big_tf_unet_2 import view

    ws = _workspace(tmp_path)
    monkeypatch.chdir(ws)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "100_step_3_denoise.out").write_text("older\n")
    (logs / "200_0_step_3_denoise.out").write_text("newer\n")
    (logs / "300_step_6_refine.out").write_text("other step\n")
    found = [p.name for p in view.here().logs_for("step_3")]
    assert found == ["100_step_3_denoise.out", "200_0_step_3_denoise.out"]


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
# step_0 dead-modality guard (the co2 case)
# ---------------------------------------------------------------------------


def test_zero_window_modality_raises(tmp_path):
    """An all-zero diagnostic must fail in step_0, not silently reach step_7."""
    pytest.importorskip("h5py")
    pytest.importorskip("torch")
    intake = pytest.importorskip("tokeye.training.big_tf_unet_2.step_0_intake")
    np = pytest.importorskip("numpy")
    h5py = pytest.importorskip("h5py")

    foundation = tmp_path / "foundation"
    foundation.mkdir()
    with h5py.File(foundation / "1001_processed.h5", "w") as f:
        grp = f.create_group("co2")
        grp.create_dataset("ydata", data=np.zeros((4, 200_000), dtype=np.float32))
        grp.create_dataset(
            "xdata", data=np.arange(200_000, dtype=np.float64) / 500_000.0
        )
    shots = tmp_path / "shots.txt"
    shots.write_text("1001\n")

    settings = {
        "shots_path": shots,
        "foundation_dir": foundation,
        "modality": "co2",
        "input_key": "co2",
        "channels": [0, 1, 2, 3],
        "subseq_len": 66_000,
        "preemphasis_coeff": 0.99,
        "target_rate_khz": 500,
        "max_windows_per_shot": 2,
        "n_shots": None,
        "out_h5": tmp_path / "step_0.h5",
        "frame_info_csv": tmp_path / "frame_info_raw.csv",
        "run_id": "t",
        "smoke": False,
    }
    with pytest.raises(RuntimeError, match="all-zero"):
        intake.main(settings)
