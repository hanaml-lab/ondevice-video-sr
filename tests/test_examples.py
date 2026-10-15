import importlib.util
from dataclasses import asdict
from pathlib import Path

import torch

from drift_vsr.train import Config, train

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upscale_clip_with_a_saved_student(tmp_path):
    cfg = Config(iters_fp=10, iters_qat=10, eval_frames=20, video_frames=24, threads=2)
    res = train(cfg, log=lambda s: None)
    torch.save({"config": asdict(cfg), "state_dict": res["model"].state_dict()}, tmp_path / "student.pt")
    data = ROOT / "examples" / "data"
    out = _example("upscale_clip").main([str(data / "clip_lr.npy"), "--checkpoint", str(tmp_path / "student.pt"),
                                         "--reference", str(data / "clip_hr.npy"), "--out", str(tmp_path / "up.npy")])
    assert len(out["relative_state_error"]) == 40 and len(out["sigma"]) == 20
    assert out["psnr"] > 15 and (tmp_path / "up.npy").exists()
