#!/usr/bin/env python3
"""Host-side restart interlock for the shared A100 pipeline."""

import argparse
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


LIMIT = 3
STABLE_US = 120_000_000
EPOCH = datetime.fromisoformat("1970-01-01T00:00:00+00:00")
ATTEMPT = re.compile(r"(\d+)-\d+\.log")
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


def interrupted_run(path, started, finished):
    with path.open(errors="replace") as log:
        header = log.readline().rstrip("\n")
        if not header.startswith("EVACUATION_ATTEMPT "):
            return None
        begin = None
        for line in log:
            if line.startswith("EVACUATION_TRAIN_BEGIN "):
                begin = int(line.split()[1])
            elif line.startswith(("EVACUATION_TRAIN_END", "EVACUATION_STAGE_EXIT")):
                return None
            elif not WARNING.match(line) and TECHNICAL.search(line):
                return None
        if begin is not None and started <= begin <= finished - STABLE_US:
            return header.split(" ", 1)[1]
    return None


def observe(directory, state, snapshot):
    if not snapshot or snapshot["Status"] not in ("exited", "dead"):
        return False
    started_at, finished_at = snapshot["StartedAt"], snapshot["FinishedAt"]
    event = [started_at, finished_at]
    started, finished = docker_us(started_at), docker_us(finished_at)
    if (finished - started < STABLE_US or snapshot["OOMKilled"] or snapshot["Error"]
            or started_at in state["intentional_stops"]):
        return False

    changed = False
    runs = state["runs"]
    for path in (directory / "evacuation-attempts").glob("*.log"):
        match = ATTEMPT.fullmatch(path.name)
        if not match or not started <= int(match[1]) <= finished - STABLE_US:
            continue
        run = interrupted_run(path, started, finished)
        if run and event not in runs.setdefault(run, []):
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
        if observe(args.directory, state, snapshot):
            save(path, state)
        if args.action == "observe":
            return 0
        running = snapshot and snapshot["Status"] == "running"
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
