"""Offline regression checks for the published ProactiveFrame experiment preset."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_stream_eval as runtime
import run_stream_eval_parallel as parallel

SCRIPT = ROOT / "scripts" / "eval_qwen3_vl_8b_proactive_frame.sh"
CONFIG = ROOT / "configs" / "proactive_frame.yaml"


class ProactiveFramePresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fastbench-preset-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        # Capture the actual Python invocation without running a model or judge.
        stub = self.work / "python"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'env': {\n"
            "k: v for k, v in os.environ.items() if k.startswith('STREAMEVAL_')\n"
            "}}))\n"
        )
        stub.chmod(0o755)
        names = {
            "MODEL_PATH", "NUM_GPUS", "NUM_WORKERS", "NUM_SCORERS", "BASE_PORT",
            "HOST", "VIDEO_ROOT", "VIDEO_ROOT_OVERRIDE", "CONFIG_PATH", "OUTPUT_DIR",
            "RUN_ID", "STREAM_ADDR_ROOT", "CHUNK_CACHE_ROOT",
        }
        self.env = {
            k: v for k, v in os.environ.items()
            if k not in names and not k.startswith("STREAMEVAL_")
        }
        self.env["PATH"] = str(self.work) + os.pathsep + self.env.get("PATH", "")

    def invoke(self, *args, env=None, check=True):
        return subprocess.run(
            ["bash", str(SCRIPT), *args], cwd=self.work,
            env={**self.env, **(env or {})}, text=True, capture_output=True, check=check,
        )

    def capture(self, *args, env=None):
        captured = json.loads(self.invoke(*args, env=env).stdout)
        with patch.object(sys, "argv", captured["argv"]):
            parsed = parallel.parse_args()
        return parsed, captured["env"]

    def test_full_method_preset_and_dedicated_config(self):
        args, env = self.capture()
        self.assertEqual(args.model_name, "qwen3_vl")
        self.assertEqual(Path(args.config), CONFIG)
        self.assertEqual(Path(args.video_root), ROOT)
        self.assertEqual((args.trim_fps, args.model_video_fps), (2.0, 2.0))
        self.assertTrue(args.proactive_focus)
        self.assertTrue(args.time_compress)
        self.assertTrue(args.low_fps_degeneration)
        self.assertEqual(args.max_focus_context_frames, 90)
        self.assertEqual(args.pixel_budget, 248832000)
        self.assertEqual(args.active_window, 4)
        self.assertEqual(args.time_window, 4.0)
        self.assertFalse(args.force_focus)
        self.assertFalse(args.full_video_high_fps)
        self.assertFalse(args.allow_early_correct)
        self.assertFalse(args.resolution_compress)
        self.assertFalse(args.skip_scoring)
        self.assertIsNone(args.visual_token_budget)
        self.assertEqual((args.num_workers, args.num_scorers), (8, 20))
        self.assertEqual(args.api_bases.split(","), [
            f"http://127.0.0.1:{8000 + i}/v1" for i in range(8)
        ])
        self.assertTrue(args.run_id.startswith("qwen3-vl-8b-proactive-frame-"))
        self.assertEqual(env["STREAMEVAL_JUDGER_BACKEND"], "openrouter")
        self.assertEqual(env["STREAMEVAL_JUDGER_MODEL"], "qwen/qwen3-235b-a22b-2507")
        self.assertNotIn("STREAMEVAL_JUDGER_API_KEY", env)

    def test_original_optional_controls_remain_available(self):
        args, _ = self.capture(
            "--num-samples", "5", "--run-id", "resume-test", "--resume",
            "--force-focus", "--allow-early-correct", "--active-window", "6",
            "--time-window-seconds", "3.5",
        )
        self.assertTrue(args.resume)
        self.assertTrue(args.force_focus)
        self.assertTrue(args.allow_early_correct)
        self.assertEqual(args.max_samples, 5)
        self.assertEqual(args.focus_window_seconds, 3.0)
        self.assertEqual(args.time_window, 3.5)
        self.assertEqual(args.active_window, 6)

    def test_paths_with_spaces_and_config_override(self):
        custom_config = self.work / "method config.yaml"
        custom_config.write_bytes(CONFIG.read_bytes())
        args, env = self.capture(
            "--model-path", "/models/my Qwen", "--video-root", "/data/video root",
            "--output-dir", "/data/run results", "--config", str(custom_config),
            "--run-id", "path-test", "--num-gpus", "2", "--base-port", "9100",
            "--num-workers", "3", "--num-scorers", "4",
        )
        self.assertEqual(args.config, str(custom_config))
        self.assertEqual(args.video_root, "/data/video root")
        self.assertEqual(args.output_dir, "/data/run results")
        self.assertEqual(env["STREAMEVAL_QWEN3_VL_MODEL"], "/models/my Qwen")
        self.assertEqual(args.api_bases, "http://127.0.0.1:9100/v1,http://127.0.0.1:9101/v1")
        self.assertEqual((args.num_workers, args.num_scorers), (3, 4))

    def test_environment_overrides_and_inference_only(self):
        args, env = self.capture("--skip-scoring", env={
            "NUM_GPUS": "1", "NUM_WORKERS": "1", "NUM_SCORERS": "2",
            "MODEL_PATH": "/checkpoints/qwen", "HOST": "localhost", "BASE_PORT": "9000",
            "VIDEO_ROOT_OVERRIDE": "/legacy/root", "CHUNK_CACHE_ROOT": "/tmp/chunk cache",
            "STREAM_ADDR_ROOT": "/tmp/stream cache", "RUN_ID": "environment-test",
            "STREAMEVAL_JUDGER_BACKEND": "openai_compatible",
            "STREAMEVAL_JUDGER_API_BASE": "http://localhost:9900/v1",
            "STREAMEVAL_JUDGER_MODEL": "test-judge",
            "STREAMEVAL_JUDGER_API_KEY": "test-key",
        })
        self.assertTrue(args.skip_scoring)
        self.assertEqual(args.api_bases, "http://localhost:9000/v1")
        self.assertEqual(args.video_root, "/legacy/root")
        self.assertEqual(args.chunk_cache_root, "/tmp/chunk cache")
        self.assertEqual(args.stream_addr_root, "/tmp/stream cache")
        self.assertEqual(args.run_id, "environment-test")
        self.assertEqual(env["STREAMEVAL_JUDGER_BACKEND"], "openai_compatible")
        self.assertEqual(env["STREAMEVAL_JUDGER_API_BASE"], "http://localhost:9900/v1")
        self.assertEqual(env["STREAMEVAL_JUDGER_MODEL"], "test-judge")
        self.assertEqual(env["STREAMEVAL_JUDGER_API_KEY"], "test-key")

    def test_rejects_invalid_arguments_before_python(self):
        for options in [
            ["--num-gpus", "0"], ["--num-workers", "-1"], ["--num-samples", "bad"],
            ["--active-window", "1.5"], ["--time-window", "-2"],
            ["--focus-window-seconds", "-1"], ["--base-port", "65535"],
            ["--model-path"], ["--config", "/missing/config.yaml"], ["--unknown"],
        ]:
            with self.subTest(options=options):
                result = self.invoke(*options, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"argv"', result.stdout)

    def test_dry_run_does_not_invoke_python_or_print_credentials(self):
        result = self.invoke("--dry-run", env={"STREAMEVAL_JUDGER_API_KEY": "test-secret"})
        tokens = shlex.split(result.stdout)
        self.assertEqual(tokens[0], "python")
        self.assertIn(str(CONFIG), tokens)
        self.assertNotIn("test-secret", result.stdout + result.stderr)
        self.assertNotIn('"argv"', result.stdout)

    def test_dedicated_prompt_matches_original_research_prompt(self):
        config = yaml.safe_load(CONFIG.read_text())
        prompt = config["prompts"]["streaming"]["system_prompt"]
        # Snapshot of the original research prompt, including all whitespace.
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),
                         "ffa7b4981e3dc042dfa9b945943b531f1bdb8261f73b1195d4df885faccb0938")
        baseline = yaml.safe_load((ROOT / "config.yaml").read_text())
        for key in ["benchmarks", "models", "judger"]:
            self.assertEqual(config[key], baseline[key])
        self.assertNotEqual(prompt, baseline["prompts"]["streaming"]["system_prompt"])

    def test_runtime_filters_focus_tokens_and_multilingual_acknowledgements(self):
        config = yaml.safe_load(CONFIG.read_text())
        baseline = yaml.safe_load((ROOT / "config.yaml").read_text())
        placeholders = baseline["scoring"]["placeholder_responses"]
        self.assertEqual(config["scoring"]["placeholder_responses"],
                         placeholders[:2] + ["focus_start", "focus_end"] + placeholders[2:])
        runtime.init_runtime_constants(config)
        self.addCleanup(runtime.init_runtime_constants, baseline)
        for value in config["scoring"]["placeholder_responses"] + ["Focus_Start", "Focus_End"]:
            with self.subTest(response=value):
                self.assertTrue(runtime.is_placeholder(value))
        self.assertFalse(runtime.is_placeholder("The ball bounced twice."))


if __name__ == "__main__":
    unittest.main()
