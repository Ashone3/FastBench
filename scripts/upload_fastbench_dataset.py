#!/usr/bin/env python3
"""Validate and upload only the FastBench dataset card, annotations, and videos."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re


ANNOTATION_NAME = "proactive_perception_annotations.json"
PRIVATE_KEYS = {"annotation_status", "remap_meta"}
LOCAL_PATH = re.compile(r"(?:/(?:mnt|home|Users|tmp|root|data|workspace)/|[A-Za-z]:[\\/])")


def validate_metadata(value):
    """Reject known private metadata and common machine-local path strings."""
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("Remove annotation_status and remap_meta before uploading.")
        for item in value.values():
            validate_metadata(item)
    elif isinstance(value, list):
        for item in value:
            validate_metadata(item)
    elif isinstance(value, str) and (value.startswith(("/", "~", "\\\\")) or LOCAL_PATH.search(value)):
        raise ValueError("Annotations contain a possible local absolute path; inspect them before uploading.")


def checked_file(root, relative):
    """Require an ordinary file within the dataset directory, without symlinks."""
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed in upload files: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside the dataset directory: {relative}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty upload file: {relative}")
    return path


def build_upload_manifest(dataset_dir):
    """Return an explicit file allowlist; unrelated local files are never selected."""
    root = Path(dataset_dir).resolve()
    annotation = checked_file(root, Path(ANNOTATION_NAME))
    card = checked_file(root, Path("README.md"))
    records = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Annotations must be a nonempty list of video records.")
    validate_metadata(records)
    videos = {}
    qa_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"video_path", "verified_responses"}:
            raise ValueError(f"Record {index} must contain only video_path and verified_responses.")
        raw_path = record["video_path"]
        if not isinstance(raw_path, str):
            raise ValueError(f"Record {index} has a non-string video_path.")
        parts = raw_path.split("/")
        if (
            len(parts) < 4
            or parts[:2] != ["dataset", "qa_video"]
            or any(part in ("", ".", "..") for part in parts)
            or "\\" in raw_path
            or PurePosixPath(raw_path).suffix != ".mp4"
        ):
            raise ValueError(f"Record {index} needs a relative dataset/qa_video/.../*.mp4 path.")
        relative = Path(*parts[1:])
        videos[relative.as_posix()] = checked_file(root, relative)
        responses = record["verified_responses"]
        if not isinstance(responses, list) or not responses or not all(isinstance(q, dict) for q in responses):
            raise ValueError(f"Record {index} needs a nonempty verified_responses list of objects.")
        qa_count += len(responses)
    files = [("README.md", card), (ANNOTATION_NAME, annotation), *sorted(videos.items())]
    return files, len(records), qa_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Hugging Face dataset ID, e.g. your-account/FastBench")
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Directory containing README.md, the annotation JSON, and qa_video/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize files without network access")
    parser.add_argument("--public", action="store_true", help="Create a new public repository (default: private)")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required for upload")
    if args.repo_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id):
        parser.error("--repo-id must have the form owner/dataset-name")
    try:
        files, record_count, qa_count = build_upload_manifest(args.dataset_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    total_bytes = sum(path.stat().st_size for _, path in files)
    print(f"Validated {record_count} video records and {qa_count} QA pairs.")
    print(f"Upload selection: README.md, {ANNOTATION_NAME}, {len(files) - 2} referenced MP4 files.")
    print(f"Total: {len(files)} files, {total_bytes / 1024 ** 3:.2f} GiB. Other local files are excluded.")
    if args.dry_run:
        print("Dry run complete. No network requests or uploads were made.")
        return 0
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        parser.error("Install the uploader dependency: python -m pip install --upgrade huggingface_hub")
    api = HfApi()  # Uses HF_TOKEN or the token stored by `hf auth login`.
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files]
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Upload FastBench annotations, videos, and dataset card",
    )
    print(f"Upload complete: {result.commit_url}")
    print("Existing repository visibility and unrelated remote files were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
