"""CLI tests for the mode-analysis suite subcommands."""

from __future__ import annotations

from tokeye.cli import build_parser, main


def test_elmspec_defaults():
    args = build_parser().parse_args(["elmspec", "input.npy"])
    assert args.command == "elmspec"
    assert args.inputs == ["input.npy"]
    assert args.model is None
    assert args.output_dir == "tokeye_elms"
    assert args.threshold == 0.5
    assert args.activity_min == 0.1
    assert args.min_gap_cols == 3
    assert args.min_duration_cols == 1
    assert args.fs is None
    assert args.png is False


def test_alfvenspec_defaults():
    args = build_parser().parse_args(["alfvenspec", "input.npy"])
    assert args.command == "alfvenspec"
    assert args.model == "ae_tf_maskrcnn"
    assert args.output_dir == "tokeye_ae"
    assert args.score_min == 0.5
    assert args.mean is None
    assert args.std is None
    assert args.save_masks is True


def test_modesearch_prints_plan_and_exits_0(capsys):
    exit_code = main(["modesearch"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "not implemented yet" in out
    assert "crawler" in out


def test_elmspec_missing_input_exits_2(tmp_path, capsys):
    exit_code = main(["elmspec", str(tmp_path / "nope_*.npy")])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "No input files found" in err
    assert "tokeye example" in err
