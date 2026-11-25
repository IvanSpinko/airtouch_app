"""Build configuration checks (PyInstaller itself is not run here)."""

import pytest

from gesture_control import config

build = pytest.importorskip("scripts.build")


def test_args_bundle_models_and_exclude_training(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "make_icon", lambda path: tmp_path / "icon.ico")
    args = build.pyinstaller_args(onedir=False)
    joined = " ".join(args)
    assert "--windowed" in args and "--onefile" in args
    assert f"{config.HAND_MODEL_PATH};models" in joined
    assert f"{config.GESTURE_MODEL_NPZ};models" in joined
    excluded = [args[i + 1] for i, a in enumerate(args) if a == "--exclude-module"]
    assert "torch" in excluded and "ml" in excluded
    assert args[args.index("--name") + 1] == config.APP_NAME


def test_missing_model_stops_the_build(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GESTURE_MODEL_NPZ", tmp_path / "missing.npz")
    monkeypatch.setattr(build, "make_icon", lambda path: tmp_path / "icon.ico")
    with pytest.raises(SystemExit):
        build.pyinstaller_args(onedir=False)


def test_exe_path():
    assert build.exe_path(onedir=False).name == f"{config.APP_NAME}.exe"
    assert build.exe_path(onedir=True).parent.name == config.APP_NAME
