# FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?

Evaluation code for **FastBench**, a benchmark for high-dynamic perception in streaming vision-language models, and **ProactiveFrame**, our training-free adaptive frame-rate baseline.

Fast events create a difficult streaming trade-off: sparse sampling can miss the evidence needed to answer a question, while dense sampling consumes the context budget and shortens the available history. FastBench evaluates whether models can capture this evidence, retain it, and answer at the appropriate time as video arrives incrementally.

This release includes local model serving, streaming inference, timing-aware scoring, and the Qwen3-VL-8B ProactiveFrame preset. The data-construction pipeline described in the paper is not bundled in this evaluation release. Model weights, benchmark annotations, and videos are obtained separately.

## FastBench at a glance

| Property | Description |
| --- | --- |
| Dataset size | 300 video clips and 306 QA pairs; six clips contain two QA pairs |
| Temporal scopes | Forward, Instant, and Backward |
| Domains | Sports, Video Games, Performing Arts, Animals, Lifestyle & Recreation, Transportation, Science & Technology, Food & Cooking |
| Capabilities | Action & Physical Interaction; Predictive & Causal Reasoning; Motion & Spatiotemporal Tracking; Entity & Visual Perception; Temporal & State Dynamics; Streaming & Online Detection |
| Streaming input | Incremental video chunks of up to one second |
| Evaluation | Answer quality and response timing under bounded visual context |

The benchmark is constructed through high-FPS QA generation, filtering out questions still answerable at 2 FPS, trajectory-based verification using SAM3 and CoTracker3, and three rounds of human inspection. The manuscript describes human-annotated evidence intervals and the full curation procedure.

## ProactiveFrame

ProactiveFrame lets the model request finer temporal observations through its textual outputs. `Focus_Start` raises the sampling rate for subsequent chunks; `Focus_End` restores sparse sampling. `Silent` leaves the current sampling mode unchanged. Decisions use the stream observed so far; the default method does not use annotated focus timestamps to trigger high-FPS observations.

A two-tier context window retains recent high-FPS observations alongside sparse history. When the focus window overflows, older focused chunks are downsampled to the base FPS. The context manager evicts older history as needed to stay within the overall budget. This preserves historical coverage while allocating more frames to transient events.

The manuscript reports an overall FastBench score of **32.9% for Qwen3-VL-8B at 2 FPS** and **38.3% with ProactiveFrame**, a gain of 5.4 percentage points. These are manuscript results, not scores recomputed during release preparation.

## Repository layout

```text
.
├── config.yaml                              # Existing baseline configuration
├── configs/proactive_frame.yaml             # Dedicated ProactiveFrame prompt and scoring configuration
├── hf_openai_server.py                      # Local model server
├── run_stream_eval.py                       # Streaming and scoring implementation
├── run_stream_eval_parallel.py              # Parallel evaluation entry point
├── scripts/
│   ├── deploy_*.sh                          # Model deployment entry points
│   ├── eval_*.sh                            # Existing baseline entry points
│   └── eval_qwen3_vl_8b_proactive_frame.sh    # Full ProactiveFrame preset
├── dataset/README.md                        # Annotation and video layout
└── tests/test_proactive_frame.py             # Offline preset checks
```

The internal benchmark key `phostream` and the `STREAMEVAL_*` environment variable names are retained for compatibility with the inherited evaluator. They do not change the benchmark's name: **FastBench**. For the same reason, the default annotation filename remains `dataset/proactive_perception_annotations.json`.

## Installation

Use Linux with a CUDA-capable GPU, Python 3.12, and `ffmpeg`/`ffprobe` on `PATH`. Install the evaluation dependencies in your selected model environment:

```bash
cd /path/to/FastBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model-specific runtime dependencies may be required by the checkpoint. If using `ATTN_IMPL=flash_attention_2`, install Flash Attention compatible with your CUDA, PyTorch, and Python versions. The existing deployment scripts default to `ATTN_IMPL=sdpa` and `TORCH_DTYPE=bf16`; use the same serving environment when comparing sampling methods.

All commands below run from the release root containing `config.yaml` and `scripts/`. In the research checkout's local release copy, use `cd /path/to/Proactive-Perception/opensource` instead. The local data already copied there remains untracked.

## Prepare FastBench data

Keep the JSON's `video_path` values unchanged. For the development annotation file, the expected layout is:

```text
.
└── dataset/
    ├── proactive_perception_annotations.json
    └── qa_video_20260921_2152_pure_en/
        ├── sample-0/original_cut.mp4
        ├── sample-1/original_cut1.mp4
        └── ...
```

From the release root, prepare a local copy from an existing dataset:

```bash
cp /path/to/proactive_perception_test_benchmark_annotations_20260923_0236_pure_en_classified_with_domains_merged_instant_primary.json \
   dataset/proactive_perception_annotations.json
cp -a /path/to/dataset/qa_video_20260921_2152_pure_en dataset/
export VIDEO_ROOT="$PWD"
```

For example, the first JSON entry uses `dataset/qa_video_20260921_2152_pure_en/sample-0/original_cut.mp4`. With this layout, `VIDEO_ROOT` is the **release root**, not `dataset/` or the sample directory. If data lives elsewhere, set `VIDEO_ROOT` to the directory containing its `dataset/` folder. Both evaluation configurations use the same annotation file.

JSON and video files are ignored by Git and are not included in the release branch. See [dataset/README.md](dataset/README.md) for more detail. A video record may contain multiple questions: `--num-samples` limits video records, not individual QA pairs.

## Deploy a model

The release provides these model entry points:

| Model | Deployment script | Baseline evaluation script | Interaction protocol |
| --- | --- | --- | --- |
| Qwen3-VL-8B | `scripts/deploy_qwen3_vl_8b.sh` | `scripts/eval_qwen3_vl_8b.sh` | Prompted streaming |
| AURA | `scripts/deploy_aura.sh` | `scripts/eval_aura.sh` | Native AURA |
| JoyAI-VL-Interaction | `scripts/deploy_joyai_vl_interaction.sh` | `scripts/eval_joyai_vl_interaction.sh` | Native JoyAI |
| MOSS-VL-Realtime | `scripts/deploy_moss_vl_realtime.sh` | `scripts/eval_moss_vl_realtime.sh` | Native MOSS-VL |
| VideoChat3-4B | `scripts/deploy_videochat3_4b.sh` | `scripts/eval_videochat3_4b.sh` | Native VideoChat3 |

Set `MODEL_PATH` to a downloaded checkpoint or its model repository identifier. For example:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export NUM_GPUS=8
export BASE_PORT=8000
bash scripts/deploy_qwen3_vl_8b.sh
```

This starts eight independent model replicas on GPUs 0 through 7 and ports 8000 through 8007. It does not split one model across eight GPUs; each replica must fit on one GPU. Keep `NUM_GPUS`, `BASE_PORT`, and `HOST` consistent between serving and evaluation. Wait for the deployment health checks to succeed before evaluating.

For a single-GPU run, set `NUM_GPUS=1`. The existing deployment entry points default to one GPU. Server logs are written to `logs/servers/`.

## Evaluate baselines

With Qwen servers running, test inference without a judge:

```bash
VIDEO_ROOT="$PWD" NUM_GPUS=8 NUM_WORKERS=8 \
bash scripts/eval_qwen3_vl_8b.sh --num-samples 1 --skip-scoring
```

For scored evaluation, configure a judge in the same shell. The original experiment uses this OpenRouter configuration:

```bash
export STREAMEVAL_JUDGER_BACKEND=openrouter
export STREAMEVAL_JUDGER_API_BASE=https://openrouter.ai/api/v1
export STREAMEVAL_JUDGER_MODEL=qwen/qwen3-235b-a22b-2507
read -rsp 'OpenRouter API key: ' STREAMEVAL_JUDGER_API_KEY; printf '\n'
export STREAMEVAL_JUDGER_API_KEY

VIDEO_ROOT="$PWD" NUM_GPUS=8 NUM_WORKERS=8 NUM_SCORERS=20 \
bash scripts/eval_qwen3_vl_8b.sh --run-id qwen3-vl-8b-baseline
```

Substitute the matching baseline evaluation script for the other models. Their native interaction flags are enabled by their existing entry points. A different OpenAI-compatible judge can be selected through the same environment variables; record that choice when comparing scores.

## Run ProactiveFrame with Qwen3-VL-8B

Use the **same deployed Qwen servers** and judge settings as above. The new entry point selects the dedicated prompt and the complete original method preset:

```bash
VIDEO_ROOT="$PWD" NUM_GPUS=8 NUM_WORKERS=8 NUM_SCORERS=20 \
bash scripts/eval_qwen3_vl_8b_proactive_frame.sh \
    --run-id qwen3-vl-8b-proactive-frame
```

For a single-GPU inference-only check:

```bash
VIDEO_ROOT="$PWD" NUM_GPUS=1 NUM_WORKERS=1 \
bash scripts/eval_qwen3_vl_8b_proactive_frame.sh \
    --num-samples 1 --skip-scoring --run-id proactive-frame-smoke
```

To inspect the exact command without inference or judge calls:

```bash
bash scripts/eval_qwen3_vl_8b_proactive_frame.sh --dry-run
```

| Setting | ProactiveFrame preset |
| --- | --- |
| System prompt / scoring config | `configs/proactive_frame.yaml` |
| Base sampling | `--trim-fps 2 --model-video-fps 2` |
| Adaptive sampling | `--proactive-focus`; native source FPS during focus |
| Focus window | `--max-focus-context-frames 90` |
| Overall source-pixel budget | `--time-compress --pixel-budget 248832000` (120 × 1920 × 1080) |
| Older focused chunks | `--low-fps-degeneration` |
| Sparse generation window | `--active-window 4` |
| Scoring window parameter | `--time-window 4.0`, with the existing task-specific timing rules |
| Early-response penalty | Enabled; `--allow-early-correct` is absent by default |
| Default concurrency | 8 servers, 8 inference workers, 20 scoring workers |

`configs/proactive_frame.yaml` preserves the research system prompt and the original multilingual `placeholder_responses`, including `focus_start`, `focus_end`, and Chinese acknowledgements. These are runtime data used to avoid treating control tokens or acknowledgements as answers. Only the annotation path is adapted to the release layout.

The public method name is **ProactiveFrame** and the script uses `proactive_frame` in its filename. The existing runtime flag `--proactive-focus` and control tokens `Focus_Start` / `Focus_End` remain unchanged. Passing only `--proactive-focus` to a baseline launcher does not select the full preset or its dedicated prompt. Use the new method entry point for this experiment.

The method entry point retains the original options, including `--active-window`, `--time-window-seconds`, `--allow-early-correct`, `--force-focus`, and `--focus-window-seconds`. `--force-focus` additionally uses annotated focus timestamps and changes the experiment into an annotation-guided ablation; it is off by default. Use `--help` for path, concurrency, and scoring options.

## Outputs and resume

Runs are saved under `runs/<model_name>_<run_id>/`, including `run_metadata.json`, `system_prompt.txt`, inference JSONL files, dialog dumps, and scoring summaries. Logs, caches, and results are ignored by Git.

To resume a method run, reuse its run ID and experimental settings:

```bash
VIDEO_ROOT="$PWD" NUM_GPUS=8 NUM_WORKERS=8 NUM_SCORERS=20 \
bash scripts/eval_qwen3_vl_8b_proactive_frame.sh \
    --run-id qwen3-vl-8b-proactive-frame --resume
```

Resume skips already saved inference records; scoring is performed unless `--skip-scoring` is set. Compare runs using the same checkpoint, serving settings, annotation version, and judge. The scorer and baseline scripts are unchanged by the new method entry point.

Stop Qwen servers when finished:

```bash
bash scripts/deploy_qwen3_vl_8b.sh stop
```

## Offline checks

The preset checks do not load model weights or contact a judge:

```bash
python -m unittest discover -s tests -p 'test_proactive_frame.py'
```

## Acknowledgements and license

The streaming evaluator builds on **PhoStream**. The original MIT copyright notice is retained in [LICENSE](LICENSE); see [NOTICE.md](NOTICE.md) for attribution. Checkpoints and benchmark media have their own terms and are not redistributed with this source code.

When using this benchmark or method, please cite the accompanying paper, **FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?**
