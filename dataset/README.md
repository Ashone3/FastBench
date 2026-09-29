# FastBench dataset setup

FastBench contains 300 video clips and 306 QA pairs. The evaluator expects an annotation JSON file and the corresponding videos; these local files are not tracked in the release branch. A video record can contain multiple questions, so `--num-samples` limits videos rather than QA pairs.

Obtain the benchmark data from the authorized project or dataset release, then place the annotation file at:

```text
dataset/proactive_perception_annotations.json
```

Each annotation contains a relative `video_path`, such as `dataset/qa_video_20260921_2152_pure_en/sample-0/original_cut.mp4`. Place that directory under the repository root, or pass a root directory with `VIDEO_ROOT=/path/to/repository` or `--video-root /path/to/repository`.

Both `config.yaml` (baselines) and `configs/proactive_frame.yaml` (ProactiveFrame) point to `dataset/proactive_perception_annotations.json`. The `phostream` key inside these configurations is an internal compatibility identifier for the inherited evaluator. The benchmark is FastBench.

The JSON's `verified_responses` array holds the questions, reference answers, temporal labels, and timestamps. Preserve these values, the optional taxonomy fields, and the relative `video_path` values when copying the data. For the layout above, run commands from the release root with `VIDEO_ROOT="$PWD"`.

The annotation records must contain the video identifier and timing fields expected by the evaluation runtime. Keep the annotation schema version aligned with the benchmark release used in your experiment.
