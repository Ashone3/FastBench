"""Offline checks for the dataset upload boundary and remote file selection."""

import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "upload_fastbench_dataset", Path(__file__).resolve().parents[1] / "scripts/upload_fastbench_dataset.py"
)
uploader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uploader)


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "qa_video/sample-0/clip.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"fixture-video")
        (self.root / "README.md").write_text("# FastBench\n")
        self.records = [{
            "video_path": "dataset/qa_video/sample-0/clip.mp4",
            "verified_responses": [{"user_query": "What happens?", "response": "A ball moves."}],
        }]
        self.save()

    def save(self):
        (self.root / uploader.ANNOTATION_NAME).write_text(json.dumps(self.records))

    def test_upload_excludes_unreferenced_files_and_uses_dataset_repo(self):
        (self.root / "private-original.json").write_text('{"secret": "local"}')
        (self.video.parent / "qa_annotation_checked.json").write_text("{}")
        (self.video.parent / "extra.mp4").write_bytes(b"not-referenced")
        api = Mock()
        operation = Mock(side_effect=lambda **kwargs: kwargs)
        fake_hub = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=operation)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            uploader.main(["--dataset-dir", str(self.root), "--repo-id", "owner/FastBench", "--public"])
        api.create_repo.assert_called_once_with(
            repo_id="owner/FastBench", repo_type="dataset", private=False, exist_ok=True
        )
        commit = api.create_commit.call_args.kwargs
        self.assertEqual(commit["repo_type"], "dataset")
        self.assertEqual(
            [op["path_in_repo"] for op in commit["operations"]],
            ["README.md", uploader.ANNOTATION_NAME, "qa_video/sample-0/clip.mp4"],
        )

    def test_dry_run_never_imports_hub(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            self.assertEqual(uploader.main(["--dataset-dir", str(self.root), "--dry-run"]), 0)

    def test_private_metadata_is_rejected_before_network(self):
        for field in ("annotation_status", "remap_meta"):
            with self.subTest(field=field):
                self.records[0]["verified_responses"][0][field] = "private"
                self.save()
                with self.assertRaises(ValueError):
                    uploader.build_upload_manifest(self.root)
                del self.records[0]["verified_responses"][0][field]

    def test_absolute_local_path_in_metadata_is_rejected(self):
        self.records[0]["verified_responses"][0]["source"] = "/mnt/private/source.mp4"
        self.save()
        with self.assertRaises(ValueError):
            uploader.build_upload_manifest(self.root)

    def test_invalid_video_paths_are_rejected(self):
        for path in ("/tmp/clip.mp4", "dataset/qa_video/../clip.mp4", "dataset/qa_video_20260921/sample-0/clip.mp4"):
            with self.subTest(path=path):
                self.records[0]["video_path"] = path
                self.save()
                with self.assertRaises(ValueError):
                    uploader.build_upload_manifest(self.root)

    def test_missing_and_symlinked_videos_are_rejected(self):
        self.video.unlink()
        with self.assertRaises(ValueError):
            uploader.build_upload_manifest(self.root)
        self.video.symlink_to(self.root / "README.md")
        with self.assertRaises(ValueError):
            uploader.build_upload_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
