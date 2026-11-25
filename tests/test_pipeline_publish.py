import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.pipeline import publish_results as publish


class PublishTests(unittest.TestCase):
    def test_auth_smoke_rejects_read_only_hf_token_before_git(self):
        api = Mock()
        api.whoami.return_value = {"auth": {"accessToken": {"role": "read"}}}
        with patch.dict(os.environ, {"HF_TOKEN": "mock-hf", "GH_TOKEN": "mock-gh"}), \
                patch.dict(sys.modules, {"huggingface_hub": Mock(HfApi=Mock(return_value=api))}), \
                patch.object(publish, "git") as git:
            with self.assertRaisesRegex(ValueError, "must have write access"):
                publish.preflight()
            api.whoami.assert_called_once()
            git.assert_not_called()

    def test_model_upload_omits_checkpoint_and_refuses_public_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = "outputs/sft-cot-test-stage2"
            out = root / path
            (out / "checkpoint-1").mkdir(parents=True)
            for f in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                      "trainer_state.json", "checkpoint-1/optimizer.pt"):
                (out / f).touch()
            api = Mock()
            with patch.object(publish, "commit") as commit, \
                    patch.object(publish.hf_sync, "ROOT", root), \
                    patch.object(publish.hf_sync, "MANIFEST", root / "hf_artifacts.json"), \
                    patch.dict(sys.modules, {"huggingface_hub": Mock(HfApi=Mock(return_value=api))}):
                api.repo_info.return_value.private = False
                with self.assertRaisesRegex(ValueError, "public repo"):
                    publish.model(path, None)
                api.upload_folder.assert_not_called()
                commit.assert_not_called()
                api.repo_info.return_value.private = True
                publish.model(path, None)
                args = api.upload_folder.call_args.kwargs
                self.assertEqual(args["repo_id"], "Shani123/re-test-sft-cot-stage2")
                self.assertEqual(args["allow_patterns"], [
                    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json"])
                self.assertEqual(publish.hf_sync._manifest()[path]["repo"], args["repo_id"])
                commit.assert_called_once()

    def test_report_only_publishes_completed_full_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = Path("outputs/eval/model-ft.test_long.preds.jsonl")
            metrics = Path(f"{result.as_posix()}.metrics.json")
            (root / result.parent).mkdir(parents=True)
            (root / result).touch()
            with patch.object(publish, "ROOT", root), \
                    patch.object(publish, "git", return_value=Mock(stdout=b"")), \
                    patch.object(publish, "commit") as commit, \
                    patch.object(publish.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "Missing predictions or metrics"):
                    publish.report("run", "model", [result.as_posix()])
                with self.assertRaisesRegex(ValueError, "Unknown evaluation"):
                    publish.report("run", "model", ["outputs/eval/smoke.model.test_long.preds.jsonl"])
                commit.assert_not_called()
                (root / metrics).touch()
                publish.report("run", "model", [result.as_posix()])
                report = Path("outputs/report/run.html")
                self.assertIn("run evaluation", (root / report).read_text(encoding="utf-8"))
                self.assertEqual([x.args[0][1] for x in run.call_args_list], [
                    "eval/append_section.py", "eval/append_errors.py"])
                self.assertEqual(set(commit.call_args.args[0]), {result, metrics, report})

    def test_git_publish_preserves_unrelated_staged_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / "origin"
            repo = root / "repo"
            subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(origin), str(repo)], check=True, capture_output=True)
            with patch.object(publish, "ROOT", repo):
                publish.git("commit", "--allow-empty", "-m", "baseline", capture_output=True)
                publish.git("push", "-u", "origin", "HEAD", capture_output=True)
                (repo / ".gitignore").write_text("outputs/\n", encoding="utf-8")
                publish.git("add", ".gitignore")
                output = Path("outputs/eval.json")
                (repo / output.parent).mkdir()
                (repo / output).write_text("{}\n", encoding="utf-8")
                publish.commit([output], "publish eval")
                committed = publish.git("show", "--format=", "--name-only", "HEAD", capture_output=True).stdout
                staged = publish.git("diff", "--cached", "--name-only", capture_output=True).stdout
                self.assertEqual(committed.decode().splitlines(), [output.as_posix()])
                self.assertEqual(staged.decode().splitlines(), [".gitignore"])


if __name__ == "__main__":
    unittest.main()
