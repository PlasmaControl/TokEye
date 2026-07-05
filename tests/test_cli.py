from __future__ import annotations

import httpx
import numpy as np
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from tokeye.cli import build_parser, main


def _repository_not_found_error() -> RepositoryNotFoundError:
    """Build a real RepositoryNotFoundError the way huggingface_hub does.

    ``HfHubHTTPError`` (its base class) requires a ``response`` kwarg with no
    default, so a bare ``RepositoryNotFoundError("msg")`` raises ``TypeError``.
    """
    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/does/not/exist")
    )
    return RepositoryNotFoundError("Repository Not Found", response=response)


class TestBuildParser:
    def test_run_subcommand_parses_options(self):
        parser = build_parser()

        args = parser.parse_args(
            ["run", "a.npy", "--hop", "128", "--no-png", "--model", "model/x.pt", "--log"]
        )

        assert args.inputs == ["a.npy"]
        assert args.hop == 128
        assert args.save_png is False
        assert args.model == "model/x.pt"
        assert args.log is True

    def test_run_subcommand_defaults(self):
        parser = build_parser()

        args = parser.parse_args(["run", "a.npy"])

        assert args.model is None  # resolved to DEFAULT_MODEL in the handler
        assert args.output_dir == "tokeye_output"
        assert args.n_fft == 1024
        assert args.hop == 256
        assert args.keep_dc is False
        assert args.clip_low == 1.0
        assert args.clip_high == 99.0
        assert args.threshold == 0.5
        assert args.save_png is True
        assert args.device == "auto"
        assert args.log is False

    def test_app_subcommand_defaults(self):
        parser = build_parser()

        args = parser.parse_args(["app"])

        assert args.port == 7860
        assert args.share is False
        assert args.open_browser is False


class TestMain:
    def test_version_returns_zero_and_prints_version(self, capsys):
        exit_code = main(["--version"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert out.strip()  # some version string was printed

    def test_no_subcommand_returns_two_or_exits(self, capsys):
        try:
            exit_code = main([])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            assert exit_code == 2

    def test_run_with_nonexistent_input_returns_two_clean_error(self, capsys):
        exit_code = main(["run", "does_not_exist_anywhere_xyz.npy"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "does_not_exist_anywhere_xyz.npy" in err
        assert "tokeye example" in err

    def test_run_with_missing_model_path_returns_two_clean_error(
        self, tmp_path, capsys
    ):
        input_path = tmp_path / "input.npy"
        np.save(input_path, np.zeros((64, 32), dtype=np.float32))

        exit_code = main(
            ["run", str(input_path), "--model", "nope/missing.pt"]
        )

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nope/missing.pt" in err
        assert "Traceback" not in err

    def test_example_writes_file_and_returns_zero(self, tmp_path, capsys):
        out_path = tmp_path / "example.npy"

        exit_code = main(
            [
                "example",
                "--output",
                str(out_path),
                "--duration",
                "0.01",
                "--fs",
                "10000",
            ]
        )

        assert exit_code == 0
        assert out_path.exists()
        sig = np.load(out_path)
        assert sig.shape[0] == 100

    def test_example_extensionless_output_prints_actual_file(self, tmp_path, capsys):
        """np.save appends .npy; the printed path must be the file that exists."""
        exit_code = main(
            [
                "example",
                "--output",
                str(tmp_path / "demo"),
                "--duration",
                "0.01",
                "--fs",
                "10000",
            ]
        )

        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        assert printed == str(tmp_path / "demo.npy")
        assert (tmp_path / "demo.npy").exists()

    def test_download_unknown_model_returns_two_clean_error(self, capsys):
        exit_code = main(["download", "not_a_real_model_name"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "not_a_real_model_name" in err

    def test_download_hub_error_returns_two_clean_error(self, monkeypatch, capsys):
        def fake_download_model(name, repo_id=None):
            raise _repository_not_found_error()

        monkeypatch.setattr("tokeye.hub.download_model", fake_download_model)

        exit_code = main(["download"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nc1/big_tf_unet" in err
        assert "TOKEYE_HF_REPO" in err
        assert "Traceback" not in err

    def test_run_hub_error_returns_two_clean_error(self, tmp_path, monkeypatch, capsys):
        input_path = tmp_path / "input.npy"
        np.save(input_path, np.zeros((64, 32), dtype=np.float32))

        def fake_load_model(source, device="auto"):
            raise _repository_not_found_error()

        monkeypatch.setattr("tokeye.hub.load_model", fake_load_model)

        exit_code = main(["run", str(input_path), "--device", "cpu"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nc1/big_tf_unet" in err
        assert "TOKEYE_HF_REPO" in err
        assert "Traceback" not in err

    def test_app_subcommand_delegates_to_app_main(self, monkeypatch):
        calls = {}

        def fake_app_main(port, share, open_browser):
            calls["port"] = port
            calls["share"] = share
            calls["open_browser"] = open_browser

        monkeypatch.setattr("tokeye.app.__main__.main", fake_app_main)

        exit_code = main(["app", "--port", "1234", "--share"])

        assert exit_code == 0
        assert calls == {"port": 1234, "share": True, "open_browser": False}


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"]])
def test_help_does_not_crash(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 0
