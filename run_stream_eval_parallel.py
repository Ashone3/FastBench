#!/usr/bin/env python3
"""
Parallel StreamEval runner.

Same semantics as run_stream_eval.py, but processes samples concurrently
via ThreadPoolExecutor.  Supports distributing requests across multiple
API endpoints (one per GPU / model-server instance).

Usage (single server, 10 concurrent workers):
    python run_stream_eval_parallel.py \
        --model-name qwen3_omni \
        --num-workers 10

Usage (10 servers on different ports, 10 workers):
    python run_stream_eval_parallel.py \
        --model-name qwen3_omni \
        --num-workers 10 \
        --api-bases http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,...
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import sqlite3
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from run_stream_eval import (
    DEFAULT_CONFIG_PATH,
    QWEN3_VL_DEFAULT_PIXEL_BUDGET,
    QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET,
    SCRIPT_DIR,
    InferenceOptions,
    UnifiedInferenceRunner,
    append_jsonl,
    build_summary,
    evaluate_sample,
    init_runtime_constants,
    load_jsonl,
    load_llm_judger,
    load_release_config,
    load_samples_any_format,
    parse_high_fps,
    resolve_model_output_input,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Runner pool – one runner per concurrent thread keeps SessionModel safe
# ---------------------------------------------------------------------------

class RunnerPool:
    """Thread-safe pool of UnifiedInferenceRunner instances."""

    def __init__(self, runners: List[UnifiedInferenceRunner]):
        self._q: queue.Queue[UnifiedInferenceRunner] = queue.Queue()
        for r in runners:
            self._q.put(r)

    def acquire(self) -> UnifiedInferenceRunner:
        return self._q.get()

    def release(self, runner: UnifiedInferenceRunner) -> None:
        self._q.put(runner)


def _sample_key(record: Dict[str, Any], bench: str) -> str:
    uuid = str(record.get("uuid", "")).strip()
    if uuid:
        return f"{bench}::uuid::{uuid}"
    sid = str(record.get("id", "")).strip()
    source = str(record.get("source", "")).strip() or str(record.get("video", "")).strip()
    return f"{bench}::id::{sid}::src::{source}"


def _build_runners(
    num_workers: int,
    config: Dict[str, Any],
    config_dir: Path,
    model_name: str,
    prompt_name: str,
    model_path: str,
    options: InferenceOptions,
    context_window_seconds: Optional[float],
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    api_bases: Optional[List[str]],
) -> List[UnifiedInferenceRunner]:
    runners: List[UnifiedInferenceRunner] = []
    for i in range(num_workers):
        cfg = copy.deepcopy(config)
        if api_bases:
            cfg["models"][model_name]["api_base"] = api_bases[i % len(api_bases)]
        runners.append(
            UnifiedInferenceRunner(
                config=cfg,
                config_dir=config_dir,
                model_name=model_name,
                prompts_name=prompt_name,
                model_path_override=model_path,
                options=options,
                context_window_seconds=context_window_seconds,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
            )
        )
    return runners


# ---------------------------------------------------------------------------
# Parallel inference
# ---------------------------------------------------------------------------

def _parallel_inference(
    pool: RunnerPool,
    num_workers: int,
    bench_tasks: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def _process(bench: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        runner = pool.acquire()
        try:
            record = runner.run_sample(sample, bench)
        finally:
            pool.release(runner)
        with write_lock:
            append_jsonl(output_dir / f"{bench}.jsonl", record)
        return record

    all_records: List[Dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(_process, bench, sample): (bench, sample)
            for bench, sample in bench_tasks
        }
        with tqdm(total=len(future_map), desc="Inference") as pbar:
            for future in as_completed(future_map):
                bench, sample = future_map[future]
                try:
                    all_records.append(future.result())
                except Exception:
                    errors += 1
                    print(
                        f"\n[ERROR] sample {sample.get('id', '?')} "
                        f"in {bench}:\n{traceback.format_exc()}"
                    )
                pbar.update(1)

    if errors:
        print(f"\n[WARN] {errors} sample(s) failed during inference")
    return all_records


# ---------------------------------------------------------------------------
# Parallel scoring
# ---------------------------------------------------------------------------

def _parallel_scoring(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    collection_name: str,
    time_window: float,
    disable_llm_judge: bool,
    num_scorers: int,
    penalize_early_response: bool,
    early_window_seconds: float,
    native_videochat3: bool = False,
    native_moss: bool = False,
    native_joyai: bool = False,
    native_aura: bool = False,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judger = None if disable_llm_judge else load_llm_judger(config)

    all_rows: List[Dict[str, Any]] = []
    rows_lock = threading.Lock()
    errors = 0

    def _score_one(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return evaluate_sample(
            sample,
            judger,
            time_window,
            penalize_early_response=penalize_early_response,
            early_window_seconds=early_window_seconds,
            native_videochat3=native_videochat3,
            native_moss=native_moss,
            native_joyai=native_joyai,
            native_aura=native_aura,
        )

    with ThreadPoolExecutor(max_workers=num_scorers) as executor:
        future_map = {executor.submit(_score_one, s): s for s in samples}
        with tqdm(total=len(future_map), desc="Scoring") as pbar:
            for future in as_completed(future_map):
                try:
                    rows = future.result()
                    with rows_lock:
                        all_rows.extend(rows)
                except Exception:
                    errors += 1
                    s = future_map[future]
                    print(
                        f"\n[ERROR] scoring sample {s.get('id', '?')}:\n"
                        f"{traceback.format_exc()}"
                    )
                pbar.update(1)

    if errors:
        print(f"\n[WARN] {errors} sample(s) failed during scoring")

    def _sample_id_sort_key(row: Dict[str, Any]) -> Tuple[int, Any]:
        sample_id = row.get("sample_id", "")
        try:
            return 0, int(sample_id)
        except (TypeError, ValueError):
            return 1, str(sample_id)

    all_rows.sort(key=_sample_id_sort_key)
    details_path = output_dir / f"{collection_name}_details.jsonl"
    write_jsonl(details_path, all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        df = pd.DataFrame(
            columns=[
                "sample_id", "question_time", "question", "answer_time",
                "answer", "response_time", "response", "score",
                "category", "task_type", "is_objective", "explanation",
            ]
        )

    summary = build_summary(df, model_name)
    summary_df = pd.DataFrame([summary]).round(1)

    db_path = output_dir / f"{collection_name}.db"
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(model_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    csv_path = output_dir / f"{collection_name}.csv"
    if csv_path.exists():
        merged = pd.concat([pd.read_csv(csv_path), summary_df], ignore_index=True)
        merged.to_csv(csv_path, index=False)
    else:
        summary_df.to_csv(csv_path, index=False)

    summary_json = output_dir / f"{collection_name}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "details_jsonl": details_path,
        "sqlite_db": db_path,
        "summary_csv": csv_path,
        "summary_json": summary_json,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel StreamEval inference + scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="", help="Optional local config yaml")
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--benchmarks", nargs="+", default=[])
    p.add_argument("--video-root", default="")
    p.add_argument("--stream-addr-root", default="")
    p.add_argument("--prompts", default="")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs"))
    p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    p.add_argument("--sparse-mode", type=int, default=1)
    p.add_argument("--active-window", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--model-video-fps", type=float, default=2.0)
    p.add_argument("--trim-fps", type=float, default=None)
    p.add_argument(
        "--force-focus",
        action="store_true",
        help="Disable chunk FPS downsampling near sample.timestamp_focus for a short window.",
    )
    p.add_argument(
        "--focus-window-seconds",
        type=float,
        default=0.0,
        help="Window size in seconds after timestamp_focus where original FPS is kept.",
    )
    p.add_argument(
        "--high-fps",
        type=parse_high_fps,
        default=None,
        metavar="FPS|original",
        help="FPS for high-FPS chunks; 'original' (default) retains source FPS.",
    )
    p.add_argument(
        "--full-video-high-fps",
        type=int,
        choices=(0, 1),
        default=0,
        metavar="{0,1}",
        help="Use high-FPS policy for every video chunk (default: 0).",
    )
    p.add_argument(
        "--proactive-focus",
        action="store_true",
        help="Allow model Focus_Start/Focus_End outputs to control high-FPS sampling.",
    )
    p.add_argument(
        "--native-videochat3",
        action="store_true",
        help=(
            "Use VideoChat3's native streaming protocol: 4 images per second, "
            "224x224 pixel budget, </Silence>/</Standby>/</Response>, and a 32-round window."
        ),
    )
    p.add_argument(
        "--native-moss",
        action="store_true",
        help=(
            "Feed each 1-second chunk through MOSS-VL's native timestamped frame sampler "
            "and realtime session. <|silence|>, <|response|>, and round tokens are scored "
            "as silence or stripped before the existing timestamp window."
        ),
    )
    p.add_argument(
        "--native-joyai",
        action="store_true",
        help=(
            "Feed each 1-second chunk as JoyAI image frames with a <T seconds> tag. "
            "</silence> is scored as silence and </response> is stripped before the "
            "existing timestamp window."
        ),
    )
    p.add_argument(
        "--native-aura",
        action="store_true",
        help=(
            "Feed each 1-second chunk as an AURA video turn. <|silent|> is scored as "
            "silence before the existing timestamp window."
        ),
    )
    p.add_argument("--chunk-cache-root", type=str, default=None)
    p.add_argument(
        "--dialog-dump-dir",
        type=str,
        default="",
        help="Directory to dump per-request prompt/images/response artifacts.",
    )
    p.add_argument("--context-window-seconds", type=float, default=None)
    p.add_argument(
        "--max-context-frames",
        type=int,
        default=120,
        help=(
            "Maximum total source frames retained across video chunks (default: 120). "
            "Set to 0 to disable this limit and use only the time window."
        ),
    )
    p.add_argument(
        "--max-focus-context-frames",
        type=int,
        default=90,
        help=(
            "Maximum high-FPS source frames retained across focus chunks (default: 90). "
            "Set to 0 to disable the dedicated focus-frame limit."
        ),
    )
    p.add_argument(
        "--resolution-compress",
        action="store_true",
        help=(
            "Keep the time window but replace total-frame eviction with dynamic "
            "resolution compression of non-focus chunks under a Qwen3-VL visual-token budget."
        ),
    )
    p.add_argument(
        "--time-compress",
        action="store_true",
        help=(
            "Keep native spatial resolution and evict the oldest video chunks "
            "when the selected visual-token or source-pixel budget is exceeded."
        ),
    )
    p.add_argument(
        "--also-compress-focus",
        action="store_true",
        help=(
            "Allow focus chunks to participate in dynamic resolution compression. "
            "Has effect only with --resolution-compress."
        ),
    )
    p.add_argument(
        "--visual-token-budget",
        type=int,
        default=None,
        help=(
            "Visual-token budget used by --resolution-compress or --time-compress. "
            f"Defaults to {QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET} unless "
            "--pixel-budget is selected."
        ),
    )
    p.add_argument(
        "--pixel-budget",
        nargs="?",
        const=QWEN3_VL_DEFAULT_PIXEL_BUDGET,
        type=int,
        default=None,
        help=(
            "Use source pixels before native Qwen preprocessing as the context "
            "budget. With no value, defaults to 120 * 1920 * 1080 = "
            f"{QWEN3_VL_DEFAULT_PIXEL_BUDGET} pixels."
        ),
    )
    p.add_argument(
        "--low-fps-degeneration",
        action="store_true",
        help=(
            "When the focus-frame window overflows, demote the oldest focus chunk "
            "to the normal model video FPS instead of discarding it."
        ),
    )

    p.add_argument("--time-window", type=float, default=2.0)
    p.add_argument(
        "--early-window-seconds",
        type=float,
        default=2.0,
        help="Allow responses up to this many seconds before answer_time (default: 2.0).",
    )
    p.add_argument(
        "--allow-early-correct",
        action="store_true",
        help="Do not force early responses to 0; score them normally if content is correct.",
    )
    p.add_argument("--collection", default="result")
    p.add_argument("--disable-llm-judge", action="store_true")

    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-scoring", action="store_true")
    p.add_argument("--model-output", default="")

    # ---- parallel-specific ----
    p.add_argument(
        "--num-workers", type=int, default=10,
        help="Number of concurrent inference workers (default: 10)",
    )
    p.add_argument(
        "--num-scorers", type=int, default=0,
        help="Number of concurrent scoring workers (default: same as --num-workers)",
    )
    p.add_argument(
        "--api-bases", default="",
        help=(
            "Comma-separated API base URLs for round-robin distribution. "
            "E.g. http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1  "
            "If empty, all workers share the config default."
        ),
    )
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Run at most N samples in this invocation (0 means all)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume inference by skipping samples already written under this run-id",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.skip_inference and not args.model_output:
        raise ValueError("--model-output is required when --skip-inference is set")
    if args.skip_inference and args.skip_scoring:
        raise ValueError("both --skip-inference and --skip-scoring are set, nothing to do")
    if args.max_context_frames < 0:
        raise ValueError("--max-context-frames must be non-negative")
    if args.max_focus_context_frames < 0:
        raise ValueError("--max-focus-context-frames must be non-negative")
    if args.visual_token_budget is not None and args.visual_token_budget <= 0:
        raise ValueError("--visual-token-budget must be positive")
    if args.pixel_budget is not None and args.pixel_budget <= 0:
        raise ValueError("--pixel-budget must be positive")
    if args.pixel_budget is not None and args.visual_token_budget is not None:
        raise ValueError("--pixel-budget and --visual-token-budget are mutually exclusive")
    if args.resolution_compress and args.time_compress:
        raise ValueError("--resolution-compress and --time-compress are mutually exclusive")
    if args.early_window_seconds < 0:
        raise ValueError("--early-window-seconds must be non-negative")
    visual_token_budget = (
        args.visual_token_budget
        if args.visual_token_budget is not None
        else QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET
    )

    num_workers = max(1, args.num_workers)
    num_scorers = max(1, args.num_scorers or num_workers)

    api_bases: Optional[List[str]] = None
    if args.api_bases.strip():
        api_bases = [u.strip() for u in args.api_bases.split(",") if u.strip()]

    config_arg = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH
    config, config_path = load_release_config(config_arg)
    init_runtime_constants(config)
    config_dir = config_path.parent if config_path.exists() else SCRIPT_DIR
    prompt_name = str(args.prompts).strip() or str(config.get("default_prompt", "streaming"))
    bench_list = args.benchmarks or list(config.get("default_benchmarks", []))
    if not bench_list and not args.skip_inference:
        raise ValueError("No benchmarks provided and no default_benchmarks found in config")

    config_video_root = str(config.get("video_root", "") or "").strip()
    config_stream_addr_root = str(config.get("stream_addr_root", "") or "").strip()
    video_root = (
        Path(args.video_root) if args.video_root
        else (Path(config_video_root) if config_video_root else None)
    )
    stream_addr_root = (
        Path(args.stream_addr_root) if args.stream_addr_root
        else (Path(config_stream_addr_root) if config_stream_addr_root else None)
    )

    run_root = Path(args.output_dir) / f"{args.model_name}_{args.run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    dialog_dump_root = Path(args.dialog_dump_dir) if args.dialog_dump_dir else (run_root / "dialog_dumps" / args.model_name)
    dialog_dump_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "config_path": str(config_path),
                "default_prompt": prompt_name,
                "created_at": datetime.now().isoformat(),
                "num_workers": num_workers,
                "num_scorers": num_scorers,
                "api_bases": api_bases,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- inference ----
    inference_samples: List[Dict[str, Any]] = []
    inference_inputs: List[Path] = []

    if not args.skip_inference:
        options = InferenceOptions(
            sparse_mode=bool(args.sparse_mode),
            active_window=args.active_window,
            max_retries=args.max_retries,
            chunk_seconds=args.chunk_seconds,
            model_video_fps=args.model_video_fps,
            trim_fps=args.trim_fps,
            force_focus=bool(args.force_focus),
            focus_window_seconds=float(args.focus_window_seconds),
            high_fps=args.high_fps,
            full_video_high_fps=bool(args.full_video_high_fps),
            max_context_frames=args.max_context_frames,
            dialog_dump_root=dialog_dump_root,
            chunk_cache_root=Path(args.chunk_cache_root) if args.chunk_cache_root else None,
            proactive_focus=bool(args.proactive_focus),
            max_focus_context_frames=args.max_focus_context_frames,
            resolution_compress=bool(args.resolution_compress),
            time_compress=bool(args.time_compress),
            also_compress_focus=bool(args.also_compress_focus),
            low_fps_degeneration=bool(args.low_fps_degeneration),
            visual_token_budget=visual_token_budget,
            pixel_budget=args.pixel_budget,
            native_videochat3=bool(args.native_videochat3),
            native_moss=bool(args.native_moss),
            native_joyai=bool(args.native_joyai),
            native_aura=bool(args.native_aura),
        )
        print(f"[Parallel] Creating {num_workers} inference runners ...")
        runners = _build_runners(
            num_workers=num_workers,
            config=config,
            config_dir=config_dir,
            model_name=args.model_name,
            prompt_name=prompt_name,
            model_path=args.model_path,
            options=options,
            context_window_seconds=args.context_window_seconds,
            video_root=video_root,
            stream_addr_root=stream_addr_root,
            api_bases=api_bases,
        )
        pool = RunnerPool(runners)

        with open(run_root / "system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(runners[0].system_prompt_text + "\n")

        bench_tasks: List[Tuple[str, Dict[str, Any]]] = []
        resumed_records: List[Dict[str, Any]] = []
        for bench in bench_list:
            bench_path = runners[0].benchmark_path(bench)
            samples = load_samples_any_format(
                benchmark_path=bench_path,
                bench_name=bench,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
                need_video_info=True,
            )
            bench_output_path = run_root / "inference" / f"{bench}.jsonl"
            done_keys = set()
            if args.resume and bench_output_path.exists():
                existing = load_jsonl(bench_output_path)
                resumed_records.extend(existing)
                for rec in existing:
                    done_keys.add(_sample_key(rec, bench))

            print(
                f"[Inference] benchmark={bench}  path={bench_path}  "
                f"samples={len(samples)}  resumed={len(done_keys)}"
            )
            for s in samples:
                if done_keys and _sample_key(s, bench) in done_keys:
                    continue
                bench_tasks.append((bench, s))

        if args.max_samples > 0:
            bench_tasks = bench_tasks[: args.max_samples]

        print(f"[Parallel] Total {len(bench_tasks)} samples × {num_workers} workers\n")

        inference_dir = run_root / "inference"
        new_records = _parallel_inference(pool, num_workers, bench_tasks, inference_dir)
        inference_samples = resumed_records + new_records

        merged_jsonl = inference_dir / "all_benchmarks.jsonl"
        write_jsonl(merged_jsonl, inference_samples)
        inference_inputs.append(merged_jsonl)
    else:
        for path in resolve_model_output_input(args.model_output):
            inference_inputs.append(path)
            inference_samples.extend(
                load_samples_any_format(
                    benchmark_path=path,
                    bench_name=path.stem,
                    video_root=video_root,
                    stream_addr_root=stream_addr_root,
                    need_video_info=False,
                )
            )

    # ---- scoring ----
    scoring_outputs: Dict[str, Path] = {}
    if not args.skip_scoring:
        print(f"\n[Parallel] Scoring with {num_scorers} workers ...")
        scoring_outputs = _parallel_scoring(
            samples=inference_samples,
            config=config,
            model_name=args.model_name,
            output_dir=run_root / "scoring",
            collection_name=args.collection,
            time_window=args.time_window,
            disable_llm_judge=args.disable_llm_judge,
            num_scorers=num_scorers,
            penalize_early_response=not args.allow_early_correct,
            early_window_seconds=args.early_window_seconds,
            native_videochat3=bool(args.native_videochat3),
            native_moss=bool(args.native_moss),
            native_joyai=bool(args.native_joyai),
            native_aura=bool(args.native_aura),
        )

    print("\n=== StreamEval Parallel Completed ===")
    print(f"Run root: {run_root}")
    print(f"Dialog dumps: {dialog_dump_root}")
    if inference_inputs:
        print("Inference inputs/outputs:")
        for path in inference_inputs:
            print(f"  - {path}")
    if scoring_outputs:
        print("Scoring outputs:")
        for key, value in scoring_outputs.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
