from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PIPELINE = Path(__file__).resolve().parents[1] / "scripts/pipeline"
BASE = 1_800_000_000
LEGACY_RUN = "qwen3.5-9b-stage2"
LEGACY_OUT = "outputs/sft-cot-qwen3.5-9b-stage2"


def docker_time(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + ".000000000Z"


def mock_host(tmp_path):
    scripts = tmp_path / "scripts/pipeline"
    scripts.mkdir(parents=True)
    for name in ("run_pipeline.sh", "gpu_watchdog.sh", "evacuation_guard.py"):
        shutil.copyfile(PIPELINE / name, scripts / name)
    # Exercise launcher ordering without sourcing real machine configs or
    # executing the production init's external exports, git, or watchdog.
    (scripts / "lib.sh").write_text(
        'LIBDIR=$PWD/scripts/pipeline; ROOT=$PWD\n'
        'init() { MACHINE=a100; RUN=$TEST_RUN; A100_DIR=$ROOT/outputs/pipeline/a100; '
        'DOCKER=docker; COMPOSE_PROJECT=test; CONTAINER=test-dev; }\n'
    )
    (scripts / "pipeline_stages.sh").write_text('echo stages >> "$DOCKER_CALLS"\n')
    (tmp_path / "docker").mkdir()
    (tmp_path / "configs/machines").mkdir(parents=True)
    (tmp_path / "configs/machines/a100.env").write_text("CONTAINER=test-dev\nDOCKER=docker\n")
    state_dir = tmp_path / "outputs/pipeline/a100"
    (state_dir / "evacuation-attempts").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("git", "setsid"):
        f = bin_dir / name
        f.write_text("#!/bin/sh\nexit 0\n")
        f.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'path = Path(os.environ["DOCKER_FIXTURE"])\n'
        'with open(os.environ["DOCKER_CALLS"], "a") as calls:\n'
        '    print(" ".join(sys.argv[1:]), file=calls)\n'
        'if sys.argv[1] == "inspect":\n'
        '    print(path.read_text())\n'
        'elif sys.argv[1] == "top":\n'
        '    assert sys.argv[2:] == ["test-dev", "-eo", "pid,ppid,args"]\n'
        '    print(path.with_name("docker.top").read_text())\n'
        'elif sys.argv[1] == "stop":\n'
        '    data = json.loads(path.read_text())\n'
        '    data[0]["State"]["Status"] = "exited"\n'
        '    path.write_text(json.dumps(data))\n'
    )
    docker.chmod(0o755)
    environment = {
        "PATH": str(bin_dir.resolve()) + os.pathsep + os.environ["PATH"],
        "DOCKER_FIXTURE": str(tmp_path / "inspect.json"),
        "DOCKER_CALLS": str(tmp_path / "docker.calls"),
        "TEST_RUN": "alpha",
    }
    return (tmp_path, scripts, state_dir), environment


def event(host, n=0, run="alpha", duration=120, log_tail="", log=True, training_started=True, **inspect):
    root, _, directory = host
    start = BASE + n * 1000
    begin = start + 40
    data = dict(Status="exited", StartedAt=docker_time(start), FinishedAt=docker_time(begin + duration),
                OOMKilled=False, Error="")
    data.update(inspect)
    (root / "inspect.json").write_text(json.dumps([{"State": data}]))
    if log:
        marker = f"EVACUATION_TRAIN_BEGIN {begin * 1_000_000}\n" if training_started else ""
        (directory / "evacuation-attempts" / f"{(start + 10) * 1_000_000}-42.log").write_text(
            f"EVACUATION_ATTEMPT {run}\n{marker}{log_tail}"
        )


def command(host, action):
    _, scripts, directory = host
    return [sys.executable, scripts / "evacuation_guard.py", directory, action, "--container", "test-dev"]


def invoke(host, action):
    return subprocess.run(command(host, action), cwd=host[0], capture_output=True, text=True, check=True)


def runs(host):
    state = host[2] / "evacuation_guard.json"
    return json.loads(state.read_text())["runs"] if state.exists() else {}


def legacy_fixture(host, n):
    root, _, directory = host
    event(host, n, log=False, Status="running")
    timestamp = datetime.fromtimestamp(BASE + n * 1000 + 1).strftime("%Y-%m-%d %H:%M:%S")
    stage = "sft-" + Path(LEGACY_OUT).name
    (directory / "pipeline.log").write_text(
        f"[{timestamp}] STAGE config INFO machine=a100 run={LEGACY_RUN} (resolved config: config.log)\n"
        f"[{timestamp}] STAGE {stage} START\n"
    )
    detail = directory / f"{stage}.log"
    detail.write_text("Starting training\n")
    os.utime(detail, (BASE + 10_000, BASE + 10_000))
    (root / "docker.top").write_text(f"PID PPID COMMAND\n42 30 /venv/bin/python train/sft_unsloth.py --out {LEGACY_OUT}\n")
    progress = root / LEGACY_OUT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "metrics.jsonl").write_text("")
    return progress


def progress_row(progress, step, elapsed, *, done=False, **metrics):
    metrics_path = progress / "metrics.jsonl"
    with metrics_path.open("a") as stream:
        stream.write(json.dumps(dict(step=step, time=elapsed, **metrics)) + "\n")
    os.utime(metrics_path, (BASE + 10_000, BASE + 10_000))
    status = json.dumps(dict(step=step, max_steps=116, done=done))
    (progress / "charts.html").write_text(f"const D=[], S={status};\n")


class EvacuationGuardTest(unittest.TestCase):
    def setUp(self):
        project = PIPELINE.parents[1]
        tmp = TemporaryDirectory(prefix=".test-evacuation-", dir=project)
        self.addCleanup(tmp.cleanup)
        self.host, environment = mock_host(Path(tmp.name))
        env_patch = patch.dict(os.environ, environment)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def observe_at(self, seconds):
        argv = command(self.host, "observe")[1:]
        spec = importlib.util.spec_from_file_location("guard", argv[0])
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with patch.object(sys, "argv", list(map(str, argv))), patch.object(guard.time, "time_ns", return_value=seconds * 10**9):
            self.assertEqual(guard.main(), 0)

    def test_legacy_requires_two_live_samples_then_counts_once(self):
        for n in range(2):
            event(self.host, n, run=LEGACY_RUN)
            invoke(self.host, "observe")
        n = 2
        start = BASE + n * 1000
        progress = legacy_fixture(self.host, n)
        docker_top = self.host[0] / "docker.top"
        trainer = docker_top.read_text().splitlines()[1].split(maxsplit=2)[2]
        docker_top.write_text(
            "PID PPID COMMAND\n30 1 micromamba run -p /venv python train/sft_unsloth.py\n"
            + "".join(f"{pid} 42 {trainer}\n" for pid in range(50, 59))
            + f"59 50 {trainer}\n42 30 {trainer}\n"
        )
        progress_row(progress, 2, 20, loss=1.0)
        self.observe_at(start + 40)
        progress_row(progress, 4, 140, loss=0.9)
        self.observe_at(start + 159)
        state = self.host[2] / "evacuation_guard.json"
        self.assertNotIn("qualified", json.loads(state.read_text())["legacy"])
        docker_top.write_text(docker_top.read_text().replace("59 50 ", "60 50 "))
        self.observe_at(start + 160)
        self.assertEqual(json.loads(state.read_text())["legacy"]["identity"]["pid"], "42")
        self.assertEqual(len(runs(self.host)[LEGACY_RUN]), 2)

        # The loop ended and the adapter was written, but the stage was still saving.
        progress_row(progress, 116, 150, done=True, train_runtime=150)
        (progress.parent / "adapter_model.safetensors").touch()
        with next(self.host[2].glob("sft-*.log")).open("a") as log:
            log.write("peak_vram_alloc_GiB=29.0\nprogress: stage2/progress/charts.html\n")
        detail = next(self.host[2].glob("sft-*.log"))
        os.utime(detail, (BASE + 10_000, BASE + 10_000))
        event(self.host, n, duration=140, log=False)
        result = subprocess.run(command(self.host, "check") + ["--run", LEGACY_RUN], cwd=self.host[0],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        invoke(self.host, "observe")
        self.assertEqual(len(runs(self.host)[LEGACY_RUN]), 3)

    def test_legacy_rejects_stale_progress_eval_and_wrong_process(self):
        for n, mode in enumerate(("already stopped", "unchanged", "eval", "smoke", "replaced trainer", "two roots")):
            with self.subTest(mode=mode):
                start = BASE + n * 1000
                progress = legacy_fixture(self.host, n)
                progress_row(progress, 2, 20, loss=1.0)
                if mode != "already stopped":
                    self.observe_at(start + 40)
                    if mode != "unchanged":
                        metrics = {"eval_loss": 0.9} if mode == "eval" else {"loss": 0.9}
                        progress_row(progress, 4, 140, **metrics)
                    top = self.host[0] / "docker.top"
                    if mode == "smoke":
                        top.write_text(top.read_text().rstrip() + " --max_steps 3\n")
                    elif mode == "replaced trainer":
                        top.write_text(top.read_text().replace("42 ", "43 "))
                    elif mode == "two roots":
                        trainer = top.read_text().splitlines()[1].split(maxsplit=2)[2]
                        with top.open("a") as stream:
                            stream.write(f"43 31 {trainer}\n50 42 {trainer}\n51 43 {trainer}\n")
                    self.observe_at(start + 160)
                    if mode == "two roots":
                        state = self.host[2] / "evacuation_guard.json"
                        self.assertNotIn("legacy", json.loads(state.read_text()))
                event(self.host, n, duration=140, log=False)
                invoke(self.host, "observe")
                self.assertEqual(runs(self.host), {})

    def test_adopted_legacy_rechecks_completion_and_errors(self):
        for n, evidence in enumerate(("stage done", "legacy deadline",
                                      "technical error", "inspect OOM", "inspect error")):
            with self.subTest(evidence=evidence):
                start = BASE + n * 1000
                progress = legacy_fixture(self.host, n)
                progress_row(progress, 2, 20, loss=1.0)
                self.observe_at(start + 40)
                progress_row(progress, 4, 140, loss=0.9)
                self.observe_at(start + 160)
                state = self.host[2] / "evacuation_guard.json"
                self.assertIn("qualified", json.loads(state.read_text())["legacy"])
                if evidence in ("stage done", "legacy deadline"):
                    progress_row(progress, 116, 150, done=True, train_runtime=150)
                    pipeline = self.host[2] / "pipeline.log"
                    line = pipeline.read_text().splitlines()[-1]
                    with pipeline.open("a") as log:
                        if evidence == "stage done":
                            log.write(line.replace("START", "DONE") + "\n")
                        else:
                            log.write(line.split(" STAGE")[0] + " STAGE watchdog DEADLINE docker stop test-dev\n")
                elif evidence == "technical error":
                    with next(self.host[2].glob("sft-*.log")).open("a") as log:
                        log.write("RuntimeError: device-side assert triggered\n")
                    detail = next(self.host[2].glob("sft-*.log"))
                    os.utime(detail, (BASE + 10_000, BASE + 10_000))
                event(self.host, n, duration=140, log=False, OOMKilled=evidence == "inspect OOM",
                      Error="runtime failure" if evidence == "inspect error" else "")
                invoke(self.host, "observe")
                self.assertEqual(runs(self.host), {})

    def test_third_stop_blocks_launcher_before_compose(self):
        root, scripts, _ = self.host
        for n in range(2):
            event(self.host, n)
            invoke(self.host, "observe")
        event(self.host, 2, log_tail="WARNING: out of memory is a possible fallback condition\nEVACUATION_TRAIN_END\n")
        launch = ["bash", scripts / "run_pipeline.sh", "--foreground"]
        blocked = subprocess.run(launch, cwd=root, capture_output=True, text=True)
        self.assertEqual(blocked.returncode, 3, blocked.stderr)
        self.assertIn("alpha blocked", blocked.stderr)
        self.assertEqual(len(runs(self.host)["alpha"]), 3)
        self.assertNotIn("compose", (root / "docker.calls").read_text())

        with patch.dict(os.environ, {"TEST_RUN": "beta"}):
            subprocess.run(launch, cwd=root, check=True, capture_output=True)
        self.assertIn("compose -p test up -d --build", (root / "docker.calls").read_text())
        self.assertEqual(set(runs(self.host)), {"alpha"})
        event(self.host, 3, log=False, Status="running")
        repeated = subprocess.run(launch, cwd=root, capture_output=True, text=True)
        self.assertEqual(repeated.returncode, 3, repeated.stderr)

    def test_exclusions(self):
        for n, options in enumerate([
            {"duration": 119},
            {"training_started": False},
            {"OOMKilled": True},
            {"Error": "runtime failure"},
            {"log_tail": "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory.\n"},
            {"log_tail": "RuntimeError: device-side assert triggered\n"},
            {"log_tail": "EVACUATION_TRAIN_END\nEVACUATION_STAGE_EXIT 0\n"},
            {"log_tail": "EVACUATION_TRAIN_END\nOSError: no space left on device\n"},
            {"log": False},  # non-training/smoke stages do not create attempt logs
        ]):
            with self.subTest(options=options):
                event(self.host, n, **options)
                invoke(self.host, "observe")
                self.assertEqual(runs(self.host), {})

    def test_concurrent_observers_deduplicate_and_ignore_stale_log(self):
        event(self.host)
        observers = [subprocess.Popen(command(self.host, "observe"), cwd=self.host[0]) for _ in range(3)]
        self.assertEqual([proc.wait(timeout=10) for proc in observers], [0, 0, 0])
        self.assertEqual(len(runs(self.host)["alpha"]), 1)
        event(self.host, n=1, log=False)
        invoke(self.host, "observe")
        self.assertEqual(len(runs(self.host)["alpha"]), 1)
        event(self.host, n=2, run="beta")
        invoke(self.host, "observe")
        self.assertEqual({run: len(stops) for run, stops in runs(self.host).items()}, {"alpha": 1, "beta": 1})

    def test_watchdog_deadline_exempts_only_its_cleanup(self):
        root, scripts, _ = self.host
        for n, status in enumerate(("running", "exited")):
            with self.subTest(status=status):
                event(self.host, n, Status=status)
                (root / "docker.calls").write_text("")
                subprocess.run(["bash", scripts / "gpu_watchdog.sh", "--machine", "a100", "--deadline-h", "0"],
                               cwd=root, check=True, capture_output=True, text=True, timeout=10)
                invoke(self.host, "observe")
                self.assertEqual("stop test-dev" in (root / "docker.calls").read_text(), status == "running")
                self.assertEqual(bool(runs(self.host)), status == "exited")
