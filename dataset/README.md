# Dataset setup

The public code expects a benchmark annotation JSON file and the corresponding video files. They are not included in this source release because their redistribution terms are separate from the evaluation code.

Obtain the benchmark data from the authorized project or dataset release, then place the annotation file at:

```text
dataset/proactive_perception_annotations.json
```

Place videos under `videos/`, or pass another directory with `VIDEO_ROOT=/path/to/videos` or `--video-root /path/to/videos`.

The annotation records must contain the video identifier and timing fields expected by the evaluation runtime. Keep the annotation schema version aligned with the benchmark release used in your experiment.
