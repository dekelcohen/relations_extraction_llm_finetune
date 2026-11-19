from datetime import datetime, timezone
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


class EvacuationGuardTest(unittest.TestCase):
    def setUp(self):
        project = PIPELINE.parents[1]
        tmp = TemporaryDirectory(prefix=".test-evacuation-", dir=project)
        self.addCleanup(tmp.cleanup)
        self.host, environment = mock_host(Path(tmp.name))
        env_patch = patch.dict(os.environ, environment)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_third_stop_blocks_launcher_before_compose(self):
        root, scripts, _ = self.host
        for n in range(2):
            event(self.host, n)
            invoke(self.host, "observe")
        event(self.host, 2, log_tail="WARNING: out of memory is a possible fallback condition\n")
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
            {"log_tail": "EVACUATION_TRAIN_END\n"},
            {"log_tail": "EVACUATION_STAGE_EXIT 0\n"},
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
