# Proactive Perception: Public Evaluation Package

This repository is a self-contained public release candidate for evaluating local streaming vision-language models on the Proactive Perception benchmark. It includes the evaluation runtime, an OpenAI-compatible Hugging Face server, model deployment launchers, and model-specific evaluation launchers.

The release is intentionally limited to stable public entry points and does not contain credentials, model weights, or private benchmark media.

## Supported models

| Model family | Server type | Native evaluation protocol |
| --- | --- | --- |
| `qwen3_vl` | `qwen3_vl` | Standard streaming protocol |
| `aura` | `qwen3_vl` | AURA native protocol |
| `joyai_vl_interaction` | `qwen3_vl` | JoyAI native protocol |
| `moss_vl_realtime` | `moss_vl` | MOSS-VL native protocol |
| `videochat3_4b` | `videochat3` | VideoChat3 native protocol |

Model checkpoints are not included. Set `MODEL_PATH` to a Hugging Face model id or a local checkpoint directory. Check each model's own license and usage terms before redistribution.

## Installation

The public package expects Linux, Python 3.12, CUDA, and `ffmpeg` with `ffprobe` available on `PATH`.

```bash
cd /path/to/public-repository
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Flash Attention 2 can improve throughput for supported CUDA and PyTorch combinations. Install a compatible wheel separately when using `ATTN_IMPL=flash_attention_2`.

## Data layout

The release branch does not track benchmark annotations or videos. The evaluator resolves the annotation file relative to this repository, and each `video_path` in the annotation is resolved relative to `VIDEO_ROOT`.

For a local smoke test with the 300-sample annotation file used during development, copy the files into this layout:

```text
.
└── dataset/
    ├── proactive_perception_annotations.json
    └── qa_video_20260921_2152_pure_en/
        ├── sample-0/
        │   └── original_cut.mp4
        ├── sample-1/
        │   └── original_cut1.mp4
        └── ...
```

The local copy can be prepared with:

```bash
cp /path/to/proactive_perception_test_benchmark_annotations_20260923_0236_pure_en_classified_with_domains_merged_instant_primary.json \
   dataset/proactive_perception_annotations.json
cp -a /path/to/dataset/qa_video_20260921_2152_pure_en dataset/
```

Then run an inference-only sample from the repository root:

```bash
VIDEO_ROOT="$PWD" \
bash scripts/eval_qwen3_vl_8b.sh --num-samples 1 --skip-scoring
```

Use `--video-root` or `VIDEO_ROOT` when the videos are stored elsewhere. See [dataset/README.md](dataset/README.md) for the annotation contract.

## Start a local model server

The public launchers start one server per visible GPU. The default is one GPU, which is convenient for a smoke test. Increase `NUM_GPUS` when the checkpoint and hardware require it.

```bash
cd /path/to/public-repository
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
NUM_GPUS=1 \
bash scripts/deploy_qwen3_vl_8b.sh
```

The server exposes `http://127.0.0.1:8000/v1`. Stop it with:

```bash
bash scripts/deploy_qwen3_vl_8b.sh stop
```

The other model launchers are:

```text
scripts/deploy_aura.sh
scripts/deploy_joyai_vl_interaction.sh
scripts/deploy_moss_vl_realtime.sh
scripts/deploy_videochat3_4b.sh
```

For AURA, JoyAI, and MOSS-VL, set `MODEL_PATH` to the checkpoint you are licensed to use. VideoChat3 defaults to `MCG-NJU/VideoChat3-4B`.

## Run evaluation

For an inference-only smoke test, no judge API is needed:

```bash
cd /path/to/public-repository
VIDEO_ROOT=/path/to/videos \
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
NUM_GPUS=1 \
bash scripts/eval_qwen3_vl_8b.sh --num-samples 1 --skip-scoring
```

For scored evaluation, configure an OpenAI-compatible judge endpoint through environment variables:

```bash
export STREAMEVAL_JUDGER_BACKEND=openai_compatible
export STREAMEVAL_JUDGER_API_BASE=https://your-judge.example.com/v1
export STREAMEVAL_JUDGER_API_KEY=your-judge-key
export STREAMEVAL_JUDGER_MODEL=your-judge-model

VIDEO_ROOT=/path/to/videos \
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
bash scripts/eval_qwen3_vl_8b.sh --num-samples 10
```

The corresponding evaluation launchers are:

```text
scripts/eval_aura.sh
scripts/eval_joyai_vl_interaction.sh
scripts/eval_moss_vl_realtime.sh
scripts/eval_videochat3_4b.sh
```

Native protocol flags are selected automatically by these launchers. Common options include `--num-gpus`, `--num-workers`, `--num-samples`, `--run-id`, `--resume`, `--force-focus`, `--proactive-focus`, and `--full-video-high-fps`. Run any launcher with `--help` for the complete list.

Results are written to `runs/`, and server logs are written to `logs/`. Both locations are ignored by Git.

## Security and release hygiene

Never commit API keys, model weights, private videos, dialog dumps, or generated results. Use environment variables or a local untracked `.env` file. Review the complete Git history for credentials before making a repository public.

## License and attribution

The evaluation code is released under the license in [LICENSE](LICENSE). See [NOTICE.md](NOTICE.md) for upstream and model attribution requirements. The benchmark data, videos, model checkpoints, and third-party dependencies may have separate terms.
