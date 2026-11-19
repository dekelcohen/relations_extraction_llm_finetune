#!/usr/bin/env python3
"""Host-side restart interlock for the shared A100 pipeline."""

import argparse
from datetime import datetime, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


LIMIT = 3
STABLE_US = 120_000_000
EPOCH = datetime.fromisoformat("1970-01-01T00:00:00+00:00")
ATTEMPT = re.compile(r"(\d+)-\d+\.log")
LEGACY_RUN = "qwen3.5-9b-stage2"
LEGACY_OUT = Path("outputs/sft-cot-qwen3.5-9b-stage2")
LEGACY_STAGE = f"sft-{LEGACY_OUT.name}"
STAGE_LINE = re.compile(r"^\[([^]]+)\] STAGE (\S+) (\S+)(?: (.*))?$")
WARNING = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)*(?:WARN(?:ING)?\b|\S+:\d+: \w*Warning:)",
    re.I,
)
TECHNICAL = re.compile(
    r"^\s*(?:[\w.]*?(?:Error|Exception|Interrupt)|SystemExit)(?::|$)"
    r"|^\s*(?:fatal|error)(?:\s|:)"
    r"|\b(?:CUDA (?:error|out of memory)|out of memory|segmentation fault|fatal Python error)\b"
    r"|^\s*\S.*:\s+(?:error:|command not found|No such file or directory)",
    re.I,
)


def docker_us(value):
    # Docker uses RFC3339Nano; inspect and the training marker share the host clock.
    value = re.sub(r"\.(\d+)", lambda m: "." + m[1][:6].ljust(6, "0"), value)
    delta = datetime.fromisoformat(value.replace("Z", "+00:00")) - EPOCH
    return delta // timedelta(microseconds=1)


def inspect(docker, container):
    result = subprocess.run([*docker, "inspect", "--type", "container", container],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        if re.search(r"\bNo such (?:object|container):", result.stderr, re.I):
            return None
        raise RuntimeError(f"Docker inspect failed: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]["State"]


def technical(line):
    return not WARNING.match(line) and TECHNICAL.search(line)


def interrupted_run(path, started, finished):
    with path.open(errors="replace") as log:
        header = log.readline().rstrip("\n")
        if not header.startswith("EVACUATION_ATTEMPT "):
            return None
        begin = None
        for line in log:
            if line.startswith("EVACUATION_TRAIN_BEGIN "):
                begin = int(line.split()[1])
            elif line.startswith("EVACUATION_STAGE_EXIT "):
                return None
            elif technical(line):
                return None
        if begin is not None and started <= begin <= finished - STABLE_US:
            return header.split(" ", 1)[1]
    return None


def legacy_files(directory, started, *, live):
    """Return evidence only for the open, uninstrumented stage-2 attempt."""
    config = opened = None
    with (directory / "pipeline.log").open(errors="replace") as log:
        for line in log:
            match = STAGE_LINE.match(line)
            if not match:
                continue
            timestamp, stage, status, detail = match.groups()
            if stage == "config":
                config = timestamp if (detail or "").split()[:2] == ["machine=a100", f"run={LEGACY_RUN}"] else None
                opened = None
            elif stage == LEGACY_STAGE and status in ("START", "DONE", "FAIL"):
                opened = timestamp if status == "START" else None
            elif stage == "pipeline" and status in ("DONE", "FAIL"):
                opened = None
            elif stage == "watchdog" and status == "DEADLINE" and opened:
                # The old watchdog did not register its intentional Docker stop.
                opened = None
    # Legacy pipeline.log uses local time with only second precision.
    if not config or not opened:
        return None
    config_us = docker_us(datetime.fromisoformat(config).astimezone().isoformat())
    stage_us = docker_us(datetime.fromisoformat(opened).astimezone().isoformat())
    if not started <= config_us <= stage_us:
        return None

    output = directory.parents[2] / LEGACY_OUT
    detail_log = directory / f"{LEGACY_STAGE}.log"
    if detail_log.is_symlink():  # New attempts have their own markers.
        return None
    metrics = output / "progress/metrics.jsonl"
    evidence = {"stage": opened, "config": config}
    for path in (detail_log, metrics):
        stat = path.stat()
        if stat.st_mtime_ns // 1000 < stage_us:
            return None
        evidence[path.name] = [stat.st_dev, stat.st_ino]
    with detail_log.open(errors="replace") as log:
        if any(technical(line) for line in log):
            return None
    latest = None
    with metrics.open() as stream:
        for line in stream:
            row = json.loads(line)
            if live and "train_runtime" in row:
                return None
            latest = row
    if latest is None or not 0 <= latest["time"] < math.inf:
        return None
    if live:
        # Require optimizer progress for adoption, not for a later stop during saves.
        with (metrics.parent / "charts.html").open(errors="replace") as chart:
            status = next((json.loads(line.rsplit(", S=", 1)[1].rstrip(";\n"))
                           for line in chart if line.startswith("const D=") and ", S=" in line), {})
        if (status.get("done") is not False or not 0 < latest["step"] == status["step"] < status["max_steps"]
                or "loss" not in latest or not math.isfinite(latest["loss"])):
            return None
    return evidence, latest


def legacy_evidence(directory, started, *, live):
    try:
        return legacy_files(directory, started, live=live)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        # Missing or partially written legacy files are not stop evidence.
        return None


def root_trainer(top):
    parents, trainers = {}, {}
    for line in top.splitlines()[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2 or not all(field.isdecimal() for field in fields[:2]):
            return None
        pid, ppid, *argv = fields
        if pid in parents:
            return None
        parents[pid] = ppid
        if (len(argv) >= 2 and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(argv[0]).name)
                and argv[1] == "train/sft_unsloth.py"):
            trainers[pid] = argv

    roots = []
    for pid, argv in trainers.items():
        # Forked workers inherit argv; independent trainers have no such ancestor.
        ancestor, visited = parents[pid], {pid}
        while ancestor in parents:
            if ancestor in visited:
                return None
            visited.add(ancestor)
            if ancestor in trainers and trainers[ancestor] == argv:
                break
            ancestor = parents[ancestor]
        else:
            roots.append(pid)
    if len(roots) == 1:
        return roots[0], trainers[roots[0]]
    return None


def legacy_sample(args, state, snapshot):
    started_at = snapshot["StartedAt"]
    if snapshot["OOMKilled"] or snapshot["Error"] or started_at in state["intentional_stops"]:
        return False
    evidence = legacy_evidence(args.directory, docker_us(started_at), live=True)
    if not evidence:
        return False
    result = subprocess.run([*args.docker, "top", args.container, "-eo", "pid,ppid,args"],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        return False
    previous = state.get("legacy")
    trainer = root_trainer(result.stdout)
    pid, argv = trainer if trainer else (None, [])
    out = argv[argv.index("--out") + 1:][:1] if "--out" in argv else []
    if (out != [str(LEGACY_OUT)]
            or any(arg.split("=")[0] in ("--max_steps", "--longest_first", "--log_dir") for arg in argv)):
        if previous and previous["identity"]["started_at"] == started_at:
            del state["legacy"]
            return True
        return False
    identity, row = evidence
    identity.update(started_at=started_at, pid=pid)
    now = time.time_ns() // 1000
    if not previous or previous["identity"] != identity or row["step"] < previous["step"]:
        state["legacy"] = {"identity": identity, "step": row["step"], "elapsed": row["time"], "seen": now}
        return True
    if (not previous.get("qualified") and now - previous["seen"] >= STABLE_US
            and row["step"] > previous["step"] and row["time"] - previous["elapsed"] >= STABLE_US / 1e6):
        previous.update(qualified=now, qualified_step=row["step"], qualified_elapsed=row["time"])
        print(f"A100 evacuation guard: observing legacy training for {LEGACY_RUN}", file=sys.stderr)
        return True
    return False


def legacy_interrupted(directory, state, started_at, started, finished):
    previous = state.get("legacy")
    if (not previous or previous["identity"]["started_at"] != started_at
            or not started <= previous.get("qualified", 0) <= finished):
        return False
    observed = legacy_evidence(directory, started, live=False)
    if not observed:
        return False
    identity, row = observed
    expected = {key: value for key, value in previous["identity"].items() if key not in ("started_at", "pid")}
    return (identity == expected and row["step"] >= previous["qualified_step"]
            and row["time"] >= previous["qualified_elapsed"])


def observe(directory, state, snapshot):
    if not snapshot or snapshot["Status"] not in ("exited", "dead"):
        return False
    started_at, finished_at = snapshot["StartedAt"], snapshot["FinishedAt"]
    event = [started_at, finished_at]
    started, finished = docker_us(started_at), docker_us(finished_at)
    if (finished - started < STABLE_US or snapshot["OOMKilled"] or snapshot["Error"]
            or started_at in state["intentional_stops"]):
        return False

    interrupted = set()
    for path in (directory / "evacuation-attempts").glob("*.log"):
        match = ATTEMPT.fullmatch(path.name)
        if not match or not started <= int(match[1]) <= finished - STABLE_US:
            continue
        run = interrupted_run(path, started, finished)
        if run:
            interrupted.add(run)
    if legacy_interrupted(directory, state, started_at, started, finished):
        interrupted.add(LEGACY_RUN)
    changed = False
    runs = state["runs"]
    for run in sorted(interrupted):
        if event not in runs.setdefault(run, []):
            runs[run].append(event)
            print(f"A100 evacuation guard: {run}: unexplained container stop "
                  f"{len(runs[run])}/{LIMIT} ({finished_at})", file=sys.stderr)
            changed = True
    return changed


def save(path, state):
    pending = path.with_suffix(".pending")
    with pending.open("w") as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_run(state, run):
    if len(state["runs"].get(run, [])) >= LIMIT:
        print(f"A100 evacuation guard: {run} blocked after {LIMIT} unexplained container stops; "
              "do not resume without explicit user instruction.", file=sys.stderr)
        return 3
    return 0


def guard(args):
    path = args.directory / "evacuation_guard.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            state = {"runs": {}, "intentional_stops": []}
        if (not isinstance(state["runs"], dict)
                or not all(isinstance(events, list) for events in state["runs"].values())
                or not isinstance(state["intentional_stops"], list)):
            raise ValueError(f"invalid state in {path}")
        if args.action == "check-state":
            return check_run(state, args.run)

        snapshot = inspect(args.docker, args.container)
        running = snapshot and snapshot["Status"] == "running"
        changed = (legacy_sample(args, state, snapshot) if running and args.action != "stop"
                   else observe(args.directory, state, snapshot))
        if changed:
            save(path, state)
        if args.action == "observe":
            return 0
        if args.action == "stop":
            if not running:
                return 0
            # Register this cleanup before stopping; the launcher cannot check or
            # replace the container while this lock is held.
            started = snapshot["StartedAt"]
            if started not in state["intentional_stops"]:
                state["intentional_stops"].append(started)
                save(path, state)
            result = subprocess.run([*args.docker, "stop", args.container], stdout=subprocess.DEVNULL)
            if result.returncode:
                state["intentional_stops"].remove(started)
                save(path, state)
            return result.returncode

        blocked = check_run(state, args.run)
        if blocked or args.action == "check" or running:
            return blocked
        return subprocess.run([*args.docker, "compose", "-p", args.project, "up", "-d", "--build"],
                              stdout=subprocess.DEVNULL).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("action", choices=("observe", "check", "check-state", "launch", "stop"))
    parser.add_argument("--run", default="")
    parser.add_argument("--container")
    parser.add_argument("--project")
    parser.add_argument("--docker", type=shlex.split, default="docker")
    args = parser.parse_args()
    try:
        return guard(args)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError, subprocess.TimeoutExpired) as exc:
        print(f"A100 evacuation guard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
