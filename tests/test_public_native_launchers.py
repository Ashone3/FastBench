"""Validate public command wiring without loading models or contacting servers."""
from pathlib import Path
import os
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def command(script, *options, extra=None):
    env = dict(os.environ)
    for name in ("EXTRA_ARGS", "MODEL_PATH", "MODEL_TYPE", "MODEL_FAMILY", "CONFIG_PATH", "VIDEO_ROOT", "NUM_GPUS", "NUM_WORKERS", "NUM_SCORERS"):
        env.pop(name, None)
    env.update(extra or {})
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--dry-run", *options],
                            cwd="/tmp", env=env, check=True, text=True, capture_output=True)
    return shlex.split(result.stdout)


@pytest.mark.parametrize("script,flag,model", [
    ("eval_moss_vl_realtime.sh", "--native-moss", "moss_vl_realtime"),
    ("eval_moss_vl_realtime_framewise.sh", "--native-moss-framewise", "moss_vl_realtime"),
    ("eval_aura.sh", "--native-aura", "aura"),
    ("eval_aurav2.sh", "--native-aura", "aura"),
    ("eval_joyai_vl_interaction.sh", "--native-joyai", "joyai_vl_interaction"),
])
def test_dedicated_entrypoints_and_legacy_protocols(script, flag, model):
    args = command(script, "--num-samples", "1")
    assert flag in args
    assert args[args.index("--model-name") + 1] == model
    assert args[args.index("--max-samples") + 1] == "1"
    assert "--allow-early-correct" not in args
    assert "--native-aurav2-prompt" not in args
    if script != "eval_moss_vl_realtime_framewise.sh":
        assert "--native-moss-framewise" not in args
    config = Path(args[args.index("--config") + 1])
    assert config == ROOT / ("configs/aurav2.yaml" if script == "eval_aurav2.sh" else "config.yaml")


def test_extra_args_preserve_quoted_paths_without_shell_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    dump = str(tmp_path / "dialog dumps")
    args = command("eval_moss_vl_realtime.sh", extra={
        "EXTRA_ARGS": f"--native-moss-framewise --dialog-dump-dir {shlex.quote(dump)} --run-id '$(touch {marker})'",
    })
    assert "--native-moss-framewise" in args
    assert args[args.index("--dialog-dump-dir") + 1] == dump
    assert not marker.exists()


def test_aurav2_config_override_and_checkpoint_environment():
    args = command("eval_aurav2.sh", "--config", "/tmp/custom.yaml", "--model-path", "/checkpoints/AURAv2")
    assert args[args.index("--config") + 1] == "/tmp/custom.yaml"


@pytest.mark.parametrize("family,kind,flag", [
    ("aurav2", "qwen3_5", "--native-aura"),
    ("aura", "qwen3_vl", "--native-aura"),
    ("moss_vl_realtime", "moss_vl", "--native-moss"),
])
def test_family_defaults_select_checkpoint_architecture(family, kind, flag):
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\n" "$MODEL_TYPE" "$NATIVE_FLAG"',
                             "test", str(ROOT / "scripts/model_defaults.sh")],
                            env={"PATH": os.environ["PATH"], "MODEL_FAMILY": family},
                            check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == [kind, flag]


def test_aurav2_respects_explicit_baseline_config_environment():
    args = command("eval_aurav2.sh", extra={"CONFIG_PATH": str(ROOT / "config.yaml")})
    assert args[args.index("--config") + 1] == str(ROOT / "config.yaml")
