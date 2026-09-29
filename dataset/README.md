# Dataset setup

The public code expects a benchmark annotation JSON file and the corresponding video files. They are not tracked in the release branch because their redistribution terms are separate from the evaluation code.

Obtain the benchmark data from the authorized project or dataset release, then place the annotation file at:

```text
dataset/proactive_perception_annotations.json
```

Each annotation contains a relative `video_path`, such as `dataset/qa_video_20260921_2152_pure_en/sample-0/original_cut.mp4`. Place that directory under the repository root, or pass a root directory with `VIDEO_ROOT=/path/to/repository` or `--video-root /path/to/repository`.

The annotation records must contain the video identifier and timing fields expected by the evaluation runtime. Keep the annotation schema version aligned with the benchmark release used in your experiment.
