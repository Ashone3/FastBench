---
language:
  - en
task_categories:
  - visual-question-answering
pretty_name: FastBench
size_categories:
  - n<1K
tags:
  - video
  - streaming-video
  - video-question-answering
  - temporal-reasoning
  - benchmark
---

# FastBench: Can Streaming VLMs Perceive High-Dynamic Real-World Streams?

FastBench evaluates high-dynamic perception in streaming vision-language models. It contains **300 video clips and 306 English question-answer pairs**. Models must observe video incrementally, capture brief events, and answer at the appropriate time.

## Benchmark coverage

| Property | Coverage |
| --- | --- |
| Temporal scopes | Forward: 168 QA pairs; Instant: 73; Backward: 65 |
| Domains | Sports; Video Games; Performing Arts; Animals; Lifestyle & Recreation; Transportation; Science & Technology; Food & Cooking |
| Capabilities | Action & Physical Interaction; Predictive & Causal Reasoning; Motion & Spatiotemporal Tracking; Entity & Visual Perception; Temporal & State Dynamics; Streaming & Online Detection |
| Evaluation | Open-ended answers with an LLM judge and task-specific timing rules |

The construction pipeline combines high-FPS QA generation, filtering out questions answerable at 2 FPS, trajectory-based verification using SAM3 and CoTracker3, and three rounds of human inspection. See the paper for the full protocol.

## Files

```text
.
├── README.md
├── proactive_perception_annotations.json
└── qa_video/
    ├── sample-0/original_cut.mp4
    ├── sample-1/original_cut1.mp4
    └── ...
```

The repository contains one annotation JSON file and 300 MP4 files, totaling approximately 8.3 GiB of video. Videos remain in their per-sample subdirectories. Local review sidecars (`qa_annotation_checked.json`) and internal metadata (`annotation_status`, `remap_meta`) are excluded. Questions, reference answers, temporal labels, and timestamps are preserved.

## Download and use with the evaluator

Download this dataset repository **into the `dataset/` directory of the FastBench code repository**. Replace `YOUR_ACCOUNT/FastBench` with this dataset's actual Hugging Face repository ID:

```bash
python -m pip install --upgrade huggingface_hub
cd /path/to/FastBench
hf download YOUR_ACCOUNT/FastBench --repo-type dataset --local-dir dataset
```

This produces `dataset/proactive_perception_annotations.json` and `dataset/qa_video/`. Annotation paths intentionally begin with `dataset/qa_video/` and resolve relative to the **code repository root**, not the annotation file's directory. Keep these paths unchanged. When running the evaluator from the code repository root, set `VIDEO_ROOT="$PWD"`.

Both `config.yaml` (baselines) and `configs/proactive_frame.yaml` (ProactiveFrame) in the code release use this annotation filename. See the code repository README for model deployment, evaluation, and judge configuration.

## Annotation format

The JSON is a list of video records. Each record contains:

| Field | Meaning |
| --- | --- |
| `video_path` | Relative video path, e.g. `dataset/qa_video/sample-0/original_cut.mp4` |
| `verified_responses` | List of QA objects associated with this clip |

Each QA object contains:

| Field | Meaning |
| --- | --- |
| `user_query` | English question presented to the model |
| `response` | Reference answer |
| `time_type` | Temporal scope: `forward`, `instant`, or `backward` |
| `timestamp_question` | Question time within the released clip |
| `timestamp_proactive` | Target response time, when provided |
| `timestamp_focus` | Annotated focus time, when provided; available for annotation-guided ablations |
| `capability` | Evaluated perception or reasoning capability |
| `domain` | Video domain label |
| `*_seconds_in_original` | Optional timing metadata in seconds on the original source timeline |

Clip timestamps are time strings (for example, `00:20`). Original-source timing metadata uses a different timeline and should not replace the clip timestamps. The evaluator determines the precise question and response schedule from the temporal scope. A limit on video records can include more QA pairs than the limit itself.

For direct inspection without a dataset loader:

```python
import json
from pathlib import Path

code_root = Path("/path/to/FastBench")
with (code_root / "dataset/proactive_perception_annotations.json").open(encoding="utf-8") as handle:
    records = json.load(handle)
video_file = code_root / records[0]["video_path"]
questions = records[0]["verified_responses"]
```

## Intended use and limitations

FastBench is intended for research evaluation of streaming video perception. Its emphasis on brief, dynamic events and its size limit how broadly scores can be generalized. Report the model checkpoint, frame-sampling policy, context budget, judge configuration, and timing protocol when comparing results. Annotated focus times should not be used to trigger observations in the default ProactiveFrame setting.

## License and attribution

The MIT license for the evaluation code does not grant rights to these videos. No dataset-wide license is declared in this card; annotation and media usage remains subject to applicable rights and source terms.
